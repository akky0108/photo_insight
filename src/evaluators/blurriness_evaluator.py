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

    def __init__(
        self,
        grad_weight: float = 0.4,
        lap_weight: float = 0.4,
        diff_weight: float = 0.2,
        t_bad: float = 50.0,
        t_poor: float = 100.0,
        t_fair: float = 200.0,
        t_good: float = 400.0,
        max_grad: float = 500.0,
        min_size: int = 64,
    ) -> None:
        self.grad_weight = float(grad_weight)
        self.lap_weight = float(lap_weight)
        self.diff_weight = float(diff_weight)

        self.t_bad = float(t_bad)
        self.t_poor = float(t_poor)
        self.t_fair = float(t_fair)
        self.t_good = float(t_good)

        self.max_grad = float(max_grad)
        self.min_size = int(min_size)

    def _to_score_and_grade(self, raw: float) -> tuple[float, str]:
        if raw <= 0:
            return 0.0, "very_blurry"

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
