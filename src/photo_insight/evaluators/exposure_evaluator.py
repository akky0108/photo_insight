# evaluators/exposure_evaluator.py

import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple


class ExposureEvaluator:
    """
    画像の平均輝度から露出を評価するクラス。

    ポリシー:
    - 生値(raw)と意味スコア(score)を分離
    - score は基本 0.0〜1.0 の 5 段階離散値（1.0, 0.75, 0.5, 0.25, 0.0）
    - 0..1 輝度に正規化してから評価（8bit/16bit/float を吸収）
    - 欠損・評価不能時はフォールバックで破綻しない
    """

    def __init__(
        self,
        lower: float = 0.14,
        upper: float = 0.55,
        margin: float = 0.10,
        fallback_score: float = 0.5,
    ) -> None:
        """
        :param lower: 0..1 の範囲で「暗めポートレート」の許容下限
        :param upper: 0..1 の範囲で「暗めポートレート」の許容上限
        :param margin: lower/upper からの許容マージン
        :param fallback_score: 評価不能時のデフォルトスコア（通常は 0.5）
        """
        self.lower = float(lower)
        self.upper = float(upper)
        self.margin = float(margin)
        self.fallback_score = float(fallback_score)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        露出を評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: 露出評価結果
            - exposure_score: 5 段階離散スコア (1.0 / 0.75 / 0.5 / 0.25 / 0.0)
            - exposure_grade: "excellent" / "good" / "fair" / "poor" / "bad"
            - mean_brightness: 0..1 の平均輝度
            - mean_brightness_8bit: 0..255 相当の平均輝度（デバッグ用）
            - exposure_eval_status: "ok" / "fallback"
            - exposure_fallback_reason: フォールバック理由 or None
            - image_dtype: 入力画像の dtype 文字列表現
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid input: expected a numpy array representing an image.")
        if image.size == 0:
            return self._fallback_result(reason="empty_image")

        # 0..1 輝度に正規化（NoiseEvaluator._to_luma01 と同系）
        luma01, dtype_info = self._to_luma01(image)

        # 平均輝度（raw）
        mean = float(np.mean(luma01))

        # 5 段階の離散スコアへ変換
        score, grade = self._score_discrete(mean)

        return {
            # --- 意味スコア（decide_accept で使う想定） ---
            "exposure_score": score,  # 1.0 / 0.75 / 0.5 / 0.25 / 0.0
            "exposure_grade": grade,  # "excellent"〜"bad"
            # --- 生値(raw) ---
            "mean_brightness": mean,  # 0..1
            "mean_brightness_8bit": mean * 255.0,  # デバッグ用
            # --- メタ情報 ---
            "exposure_eval_status": "ok",
            "exposure_fallback_reason": None,
            "image_dtype": dtype_info,
        }

    # ------------------------------------------------------------------
    # 内部ヘルパ
    # ------------------------------------------------------------------
    def _to_luma01(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        BGR/Gray + 各種 bit 深度を 0..1 の輝度に正規化して返す。

        NoiseEvaluator._to_luma01 と同じ思想:
        - uint8 なら /255
        - uint16 なら /65535
        - float 系は max 値を見て 0..1 か 0..max に正規化
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        dtype = gray.dtype
        dtype_info = str(dtype)

        g = gray.astype(np.float32)

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

    def _score_discrete(self, mean: float) -> Tuple[float, str]:
        """
        平均輝度 mean(0..1) を 5 段階の離散スコアに変換。

        方針:
        - [lower, upper] を「理想ゾーン(excellent)」
        - そこからの距離 |mean - center| に応じて 5 段階

        ゾーン例:
            - excellent:   mean ∈ [lower, upper]
            - good:        center±(half_range + 0.5*margin)
            - fair:        center±(half_range + 1.0*margin)
            - poor:        center±(half_range + 1.5*margin)
            - bad:         それ以外
        """
        lower = self.lower
        upper = self.upper
        margin = self.margin

        # guard: lower/upper が変になっていたら中央 0.5 付近を基準にする
        if not (0.0 <= lower < upper <= 1.0):
            lower = 0.3
            upper = 0.7

        center = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)

        d = abs(mean - center)

        # excellent: 理想ゾーン内
        if lower <= mean <= upper:
            return 1.0, "excellent"

        # good/fair/poor のバンド境界
        t2 = half_range + 0.5 * margin  # good
        t3 = half_range + 1.0 * margin  # fair
        t4 = half_range + 1.5 * margin  # poor

        if d <= t2:
            return 0.75, "good"
        if d <= t3:
            return 0.5, "fair"
        if d <= t4:
            return 0.25, "poor"
        return 0.0, "bad"

    def _fallback_result(
        self,
        reason: str,
        dtype_info: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        評価不能時のフォールバック結果。
        NoiseEvaluator と同様に、中間スコアをデフォルトとする。
        """
        return {
            "exposure_score": float(self.fallback_score),  # 通常は 0.5
            "exposure_grade": "fair",  # 真ん中相当
            "mean_brightness": None,
            "mean_brightness_8bit": None,
            "exposure_eval_status": "fallback",
            "exposure_fallback_reason": reason,
            "image_dtype": dtype_info,
        }
