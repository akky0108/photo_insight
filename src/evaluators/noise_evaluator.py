import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple


class NoiseEvaluator:
    """
    画像のノイズを評価するクラス。

    ポリシー:
    - 生値(raw)と意味スコア(score)を分離
    - score は基本 0.0〜1.0 の 5 段階離散値（1.0, 0.75, 0.5, 0.25, 0.0）
    - 露出・解像度・ビット深度に極力依存しない
    - 欠損・評価不能時はフォールバックで破綻しない

    契約:
    - noise_raw は「高いほど良い」で扱うため、noise_raw = -noise_sigma_used を固定する
    """

    def __init__(
        self,
        # 旧パラメータ（後方互換用に受けるだけ・内部では使わない）
        max_noise_value: float = 70.0,
        # 新パラメータ群（必要なら外から上書き）
        downsample_long_edge: int = 1024,
        gaussian_sigma: float = 1.2,
        midtone_min: float = 0.15,
        midtone_max: float = 0.75,
        grad_thr: float = 0.08,
        min_mask_ratio: float = 0.05,
        good_sigma: float = 0.010,
        warn_sigma: float = 0.018,
        fallback_mode: str = "global_mad",  # or "skip"
        fallback_score: float = 0.5,
        # 追加（統一）
        logger=None,
        config=None,
    ):
        """
        :param max_noise_value: 旧実装互換のために受け取るが内部では使用しない
        :param downsample_long_edge: 長辺をこのサイズに揃えて評価
        :param gaussian_sigma: 残差算出用ガウシアンのσ
        :param midtone_min: 評価対象とする輝度(0..1)の下限
        :param midtone_max: 評価対象とする輝度(0..1)の上限
        :param grad_thr: 勾配強度の閾値（高すぎるエッジ/テクスチャは除外）
        :param min_mask_ratio: マスク有効画素の最低割合。下回るとフォールバック
        :param good_sigma: この値付近までが「良好」ゾーンの目安
        :param warn_sigma: この値を超えると「かなりノイジー」の目安
        :param fallback_mode: マスク不足等のときの挙動
        :param fallback_score: 最終的にどうしても計算不能なときのスコア
        """
        self.logger = logger

        # 旧パラメータは残しておく（ログ用など）
        self.max_noise_value = max_noise_value

        # まずは引数の値で初期化（= デフォルト“保険”）
        self.downsample_long_edge = int(downsample_long_edge)
        self.gaussian_sigma = float(gaussian_sigma)
        self.midtone_min = float(midtone_min)
        self.midtone_max = float(midtone_max)
        self.grad_thr = float(grad_thr)
        self.min_mask_ratio = float(min_mask_ratio)
        self.good_sigma = float(good_sigma)
        self.warn_sigma = float(warn_sigma)
        self.fallback_mode = str(fallback_mode)
        self.fallback_score = float(fallback_score)

        # config で上書き（統一）
        cfg = config or {}
        noise_cfg = cfg.get("noise", {}) if isinstance(cfg, dict) else {}
        if isinstance(noise_cfg, dict):
            self._apply_config_overrides(noise_cfg)

        if self.logger is not None:
            try:
                self.logger.debug(
                    "[NoiseEvaluator] config applied: "
                    f"downsample_long_edge={self.downsample_long_edge}, "
                    f"gaussian_sigma={self.gaussian_sigma}, "
                    f"midtone_min={self.midtone_min}, midtone_max={self.midtone_max}, "
                    f"grad_thr={self.grad_thr}, min_mask_ratio={self.min_mask_ratio}, "
                    f"good_sigma={self.good_sigma}, warn_sigma={self.warn_sigma}, "
                    f"fallback_mode={self.fallback_mode}, fallback_score={self.fallback_score}"
                )
            except Exception:
                pass

    def _apply_config_overrides(self, noise_cfg: Dict[str, Any]) -> None:
        """
        config["noise"] から各種パラメータを上書きする。
        変換できない値は無視（事故防止）。
        """
        def _set_int(attr: str, key: str) -> None:
            if key not in noise_cfg:
                return
            try:
                setattr(self, attr, int(noise_cfg[key]))
            except Exception:
                return

        def _set_float(attr: str, key: str) -> None:
            if key not in noise_cfg:
                return
            try:
                setattr(self, attr, float(noise_cfg[key]))
            except Exception:
                return

        def _set_str(attr: str, key: str) -> None:
            if key not in noise_cfg:
                return
            try:
                setattr(self, attr, str(noise_cfg[key]))
            except Exception:
                return

        _set_int("downsample_long_edge", "downsample_long_edge")
        _set_float("gaussian_sigma", "gaussian_sigma")
        _set_float("midtone_min", "midtone_min")
        _set_float("midtone_max", "midtone_max")
        _set_float("grad_thr", "grad_thr")
        _set_float("min_mask_ratio", "min_mask_ratio")

        _set_float("good_sigma", "good_sigma")
        _set_float("warn_sigma", "warn_sigma")

        _set_str("fallback_mode", "fallback_mode")
        _set_float("fallback_score", "fallback_score")

        # guard（最低限の安全）
        if self.downsample_long_edge <= 0:
            self.downsample_long_edge = 1024
        if self.gaussian_sigma <= 0.0:
            self.gaussian_sigma = 1.2
        self.midtone_min = float(np.clip(self.midtone_min, 0.0, 1.0))
        self.midtone_max = float(np.clip(self.midtone_max, 0.0, 1.0))
        if self.midtone_max < self.midtone_min:
            self.midtone_min, self.midtone_max = self.midtone_max, self.midtone_min
        if self.grad_thr <= 0.0:
            self.grad_thr = 0.08
        if self.min_mask_ratio < 0.0:
            self.min_mask_ratio = 0.0
        if self.min_mask_ratio > 1.0:
            self.min_mask_ratio = 1.0

        # good/warn は壊れてても _score_discrete でガードするが、ここでも最低限
        if self.good_sigma < 0.0:
            self.good_sigma = abs(self.good_sigma)
        if self.warn_sigma < 0.0:
            self.warn_sigma = abs(self.warn_sigma)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像のノイズを評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: ノイズ評価結果
            - noise_score: 5 段階離散スコア (1.0 / 0.75 / 0.5 / 0.25 / 0.0)
            - noise_grade: "excellent" / "good" / "fair" / "poor" / "bad"
            - noise_sigma_midtone: midtone+低勾配マスク上でのロバストσ（MADベース）
            - noise_sigma_used: 実際にスコア計算に用いたσ
            - noise_mask_ratio: マスクに採用された画素比率
            - noise_raw: -noise_sigma_used（高いほど良い契約）
            - noise_eval_status: "ok" / "fallback"
            - noise_fallback_reason: フォールバック理由
            - downsampled_size: 評価に使った画像サイズ (h, w)
            - image_dtype: 入力画像のdtype文字列表現
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid input: expected a numpy array representing an image.")
        if image.size == 0:
            return self._fallback_result(reason="empty_image")

        # 1. 0..1 輝度に正規化（BGR/Gray + 8/16bit/float を吸収）
        luma01, dtype_info = self._to_luma01(image)

        # 2. 解像度を揃える
        luma01_ds, ds_size = self._downsample_long_edge(luma01, self.downsample_long_edge)

        # 3. 残差 = luma - ガウシアン平滑
        residual = self._residual(luma01_ds, self.gaussian_sigma)

        # 4. midtone + 低勾配マスクを構成
        mask = self._build_mask(luma01_ds)
        mask_ratio = float(mask.mean()) if mask.size > 0 else 0.0

        # 5. マスク上でロバストσ（MAD）を計算
        sigma_midtone: Optional[float] = None
        if mask_ratio >= self.min_mask_ratio:
            sigma_midtone = self._mad_sigma(residual[mask])

        status = "ok"
        fallback_reason = None
        sigma_used = sigma_midtone

        # マスク不足・数値異常ならフォールバック
        if sigma_used is None or not np.isfinite(sigma_used):
            status = "fallback"
            fallback_reason = "mask_too_small_or_invalid"
            if self.fallback_mode == "global_mad":
                sigma_used = self._mad_sigma(residual.reshape(-1))
                if not np.isfinite(sigma_used):
                    return self._fallback_result(
                        reason="global_mad_invalid",
                        dtype_info=dtype_info,
                        ds_size=ds_size,
                        mask_ratio=mask_ratio,
                    )
            else:  # "skip" 相当
                return self._fallback_result(
                    reason="skip",
                    dtype_info=dtype_info,
                    ds_size=ds_size,
                    mask_ratio=mask_ratio,
                )

        # 6. 5段階の離散スコアへ変換
        noise_score, grade = self._score_discrete(float(sigma_used))

        return {
            # --- 意味スコア（decide_accept で使う想定） ---
            "noise_score": float(noise_score),   # 1.0 / 0.75 / 0.5 / 0.25 / 0.0
            "noise_grade": grade,                # "excellent"〜"bad"

            # --- 生値(raw) ---
            "noise_sigma_midtone": float(sigma_midtone) if sigma_midtone is not None and np.isfinite(sigma_midtone) else None,
            "noise_sigma_used": float(sigma_used) if np.isfinite(sigma_used) else None,
            "noise_mask_ratio": mask_ratio,

            # 契約: raw は「高いほど良い」
            "noise_raw": float(-sigma_used) if (sigma_used is not None and np.isfinite(sigma_used)) else None,

            # --- メタ情報（フォールバック・条件の追跡用） ---
            "noise_eval_status": status,               # "ok" / "fallback"
            "noise_fallback_reason": fallback_reason,  # None or str
            "downsampled_size": ds_size,               # (h, w)
            "image_dtype": dtype_info,
        }

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    def _to_luma01(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        BGR/Gray + 各種bit深度を 0..1 の輝度に正規化して返す。
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        dtype = gray.dtype
        dtype_info = str(dtype)

        g = gray.astype(np.float32)

        # dtypeごとにざっくり 0..1 へ
        if np.issubdtype(dtype, np.uint8):
            g01 = g / 255.0
        elif np.issubdtype(dtype, np.uint16):
            g01 = g / 65535.0
        else:
            # floatなど：値域が 0..1 か 0..255 系か不明なので max 値を見て調整
            maxv = float(np.nanmax(g)) if g.size else 1.0
            if maxv > 1.5:
                g01 = g / maxv
            else:
                g01 = g
        g01 = np.clip(g01, 0.0, 1.0)
        return g01, dtype_info

    def _downsample_long_edge(self, img: np.ndarray, long_edge: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        長辺を long_edge に揃える（縮小のみ）。解像度依存性を減らす。
        """
        h, w = img.shape[:2]
        if max(h, w) <= long_edge:
            return img, (h, w)
        scale = long_edge / float(max(h, w))
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return resized, (nh, nw)

    def _residual(self, luma01: np.ndarray, sigma: float) -> np.ndarray:
        """
        低周波成分を引いた残差（= ノイズ＋微細テクスチャ）を算出。
        """
        base = cv2.GaussianBlur(
            luma01,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        return luma01 - base

    def _build_mask(self, luma01: np.ndarray) -> np.ndarray:
        """
        - midtone 範囲
        - 勾配が弱い画素（テクスチャ/エッジを除外）
        の AND をマスクとして返す。
        """
        mid = (luma01 >= self.midtone_min) & (luma01 <= self.midtone_max)

        gx = cv2.Sobel(luma01, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma01, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.magnitude(gx, gy)

        low_grad = grad < self.grad_thr
        return mid & low_grad

    def _mad_sigma(self, x: np.ndarray) -> float:
        """
        ロバストな標準偏差（MADベース）を計算。
        """
        if x is None or x.size < 16:
            return float("nan")
        x = x.astype(np.float32)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        # 正規分布換算
        return float(1.4826 * mad)

    def _score_discrete(self, sigma: float) -> Tuple[float, str]:
        """
        raw σ から 5 段階の離散スコアに変換。
        σ が小さいほどノイズが少ない = スコアが高い。

        閾値は good_sigma / warn_sigma をベースに自動生成:
            t1 = 0.6 * good_sigma                 → excellent
            t2 = good_sigma                       → good
            t3 = (good_sigma + warn_sigma) / 2    → fair
            t4 = warn_sigma                       → poor
            >t4                                   → bad

        good_sigma / warn_sigma の関係がおかしい場合は
        3 段階 ("good"/"warn"/"bad") にフォールバック。
        """
        g = float(self.good_sigma)
        w = float(self.warn_sigma)

        # guard: もし設定が変になっていたら旧 3 段階へフォールバック
        if not (g > 0.0 and w > g):
            if sigma <= g:
                return 1.0, "good"
            if sigma <= w:
                return 0.5, "warn"
            return 0.0, "bad"

        t1 = 0.6 * g
        t2 = g
        t3 = 0.5 * (g + w)
        t4 = w

        if sigma <= t1:
            return 1.0, "excellent"
        if sigma <= t2:
            return 0.75, "good"
        if sigma <= t3:
            return 0.5, "fair"
        if sigma <= t4:
            return 0.25, "poor"
        return 0.0, "bad"

    def _fallback_result(
        self,
        reason: str,
        dtype_info: Optional[str] = None,
        ds_size: Optional[Tuple[int, int]] = None,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        評価不能時のフォールバック結果。
        「拾う/落とす」の境界挙動を安定させるため、中間スコアをデフォルトにしている。
        """
        return {
            "noise_score": float(self.fallback_score),  # 通常は 0.5
            "noise_grade": "fair",                     # 5 段階の真ん中に相当
            "noise_sigma_midtone": None,
            "noise_sigma_used": None,
            "noise_raw": None,
            "noise_mask_ratio": float(mask_ratio) if mask_ratio is not None else None,
            "noise_eval_status": "fallback",
            "noise_fallback_reason": reason,
            "downsampled_size": ds_size,
            "image_dtype": dtype_info,
        }
