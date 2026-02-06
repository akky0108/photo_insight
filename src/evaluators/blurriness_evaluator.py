import cv2
import numpy as np
from utils.image_utils import ImageUtils


class BlurrinessEvaluator:
    """
    ぼやけ具合を評価するクラス（複数のアルゴリズムを使用）
    - blurriness_raw: 連続値。大きいほどシャープ（ピントが合っている）
    - blurriness_score: 0,0.25,0.5,0.75,1.0 の5段階
    - blurriness_grade: "very_blurry" 〜 "excellent" のラベル
    - blurriness_eval_status: "ok" / "fallback" / "invalid"
    - blurriness_fallback_reason: fallback/invalid の理由（ok は ""）
    - variance_of_* は解析用の詳細として残す
    """

    DEFAULT_DISCRETIZE_THRESHOLDS_RAW = {
        # “保険”のデフォルト（実データで調整される前提）
        "bad": 50.0,
        "poor": 100.0,
        "fair": 200.0,
        "good": 400.0,
    }

    def __init__(
        self,
        grad_weight: float = 0.4,
        lap_weight: float = 0.4,
        diff_weight: float = 0.2,
        max_grad: float = 500.0,
        min_size: int = 64,
        logger=None,
        config=None,
    ) -> None:
        self.grad_weight = float(grad_weight)
        self.lap_weight = float(lap_weight)
        self.diff_weight = float(diff_weight)

        self.max_grad = float(max_grad)
        self.min_size = int(min_size)

        self.logger = logger

        cfg = config or {}
        blur_cfg = cfg.get("blurriness", {}) if isinstance(cfg, dict) else {}

        thresholds = blur_cfg.get("discretize_thresholds_raw", {})
        if not isinstance(thresholds, dict):
            thresholds = {}

        def _get(name: str) -> float:
            v = thresholds.get(name, self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW[name])
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW[name])

        # 4境界: bad/poor/fair/good
        self.t_bad = _get("bad")
        self.t_poor = _get("poor")
        self.t_fair = _get("fair")
        self.t_good = _get("good")

        # 単調性崩れの事故防止（並べ直し）
        ts = sorted([self.t_bad, self.t_poor, self.t_fair, self.t_good])
        self.t_bad, self.t_poor, self.t_fair, self.t_good = ts

        if self.logger is not None:
            try:
                self.logger.debug(
                    f"[BlurrinessEvaluator] discretize_thresholds_raw="
                    f"bad:{self.t_bad}, poor:{self.t_poor}, fair:{self.t_fair}, good:{self.t_good}"
                )
            except Exception:
                pass

    def _to_score_and_grade(self, raw: float) -> tuple[float, str]:
        # raw: 大きいほど良い（シャープ）
        if raw <= 0 or not np.isfinite(raw):
            return 0.0, "very_blurry"

        # 閾値は「境界」なので < で段階を上げていく
        if raw < self.t_bad:
            return 0.0, "very_blurry"
        elif raw < self.t_poor:
            return 0.25, "blurry"
        elif raw < self.t_fair:
            return 0.5, "fair"
        elif raw < self.t_good:
            return 0.75, "good"
        else:
            return 1.0, "excellent"

    def _result_base(self) -> dict:
        # 欠損しない“契約”の形に固定
        return {
            "blurriness_score": 0.0,
            "blurriness_raw": 0.0,
            "blurriness_grade": "very_blurry",
            "blurriness_eval_status": "invalid",
            "blurriness_fallback_reason": "",
            # 解析用詳細（残す）
            "variance_of_gradient": 0.0,
            "variance_of_laplacian": 0.0,
            "variance_of_difference": 0.0,
        }

    def evaluate(self, image: np.ndarray) -> dict:
        result = self._result_base()

        # --- 入力チェック（invalid）---
        if not isinstance(image, np.ndarray):
            result["blurriness_eval_status"] = "invalid"
            result["blurriness_fallback_reason"] = "invalid_input_not_ndarray"
            return result

        if image.size == 0:
            result["blurriness_eval_status"] = "invalid"
            result["blurriness_fallback_reason"] = "invalid_input_empty"
            return result

        try:
            # --- グレースケール変換 ---
            gray_image = ImageUtils.to_grayscale(image) if len(image.shape) == 3 else image
            h, w = gray_image.shape[:2]

            # 小さすぎる画像は“測定不確実”として fallback（ただし計算はする）
            eval_status = "ok"
            fallback_reason = ""
            if h < self.min_size or w < self.min_size:
                eval_status = "fallback"
                fallback_reason = f"too_small_image_{w}x{h}"

            # --- Tenengrad（Sobel） ---
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            gradient_magnitude = np.clip(gradient_magnitude, 0, self.max_grad)
            variance_of_gradient = float(gradient_magnitude.var())

            # --- Laplacian ---
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=5)
            variance_of_laplacian = float(laplacian.var())

            # --- Gaussian diff ---
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            difference = cv2.absdiff(gray_image, blurred)
            variance_of_difference = float(difference.var())

            # --- 統合 raw（大きいほどシャープ） ---
            blurriness_raw = (
                self.grad_weight * variance_of_gradient
                + self.lap_weight * variance_of_laplacian
                + self.diff_weight * variance_of_difference
            )

            if not np.isfinite(blurriness_raw):
                result["blurriness_eval_status"] = "fallback"
                result["blurriness_fallback_reason"] = "non_finite_raw"
                return result

            blurriness_score, blurriness_grade = self._to_score_and_grade(float(blurriness_raw))

            # --- 反映（ok/fallback いずれでも必須キーは揃う）---
            result.update(
                {
                    "blurriness_score": float(blurriness_score),
                    "blurriness_raw": float(blurriness_raw),
                    "blurriness_grade": blurriness_grade,
                    "blurriness_eval_status": eval_status,
                    "blurriness_fallback_reason": fallback_reason,
                    "variance_of_gradient": float(variance_of_gradient),
                    "variance_of_laplacian": float(variance_of_laplacian),
                    "variance_of_difference": float(variance_of_difference),
                }
            )
            return result

        except Exception as e:
            # “悪い評価”ではなく“測定不確実”
            result["blurriness_eval_status"] = "fallback"
            result["blurriness_fallback_reason"] = f"exception:{type(e).__name__}"
            return result
