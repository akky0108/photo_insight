import cv2
import numpy as np
from utils.image_utils import ImageUtils


class BlurrinessEvaluator:
    """
    ぼやけ具合を評価するクラス（複数のアルゴリズムを使用）
    - blurriness_raw: 連続値。大きいほどシャープ（ピントが合っている）
    - blurriness_score: 0,0.25,0.5,0.75,1.0 の5段階
    - blurriness_grade: "very_blurry" 〜 "excellent" のラベル
    """

    def __init__(
        self,
        grad_weight: float = 0.4,
        lap_weight: float = 0.4,
        diff_weight: float = 0.2,
        # ↓ ここのしきい値は後で実データを見ながら調整する想定
        t_bad: float = 50.0,
        t_poor: float = 100.0,
        t_fair: float = 200.0,
        t_good: float = 400.0,
        max_grad: float = 500.0,
    ) -> None:
        self.grad_weight = grad_weight
        self.lap_weight = lap_weight
        self.diff_weight = diff_weight

        # raw スコア → 5段階スコアに変換するためのしきい値
        self.t_bad = t_bad
        self.t_poor = t_poor
        self.t_fair = t_fair
        self.t_good = t_good

        # Sobel 勾配のクリップ上限（極端なピーク対策）
        self.max_grad = max_grad

    def _to_score_and_grade(self, raw: float) -> tuple[float, str]:
        """
        連続値 blurriness_raw を 5段階スコア＋ラベルに変換する。
        raw は「大きいほどシャープ」という想定。
        """
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

    def evaluate(self, image: np.ndarray) -> dict:
        """
        画像のぼやけ具合を評価する。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: blurriness_score / blurriness_raw などを含む辞書
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(
                "Invalid input: expected a numpy array representing an image."
            )

        # グレースケール変換
        if len(image.shape) == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # Sobelエッジ検出（Tenengrad 法）
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 勾配のスケーリング（極端なピークをクリップ）
        gradient_magnitude = np.clip(gradient_magnitude, 0, self.max_grad)

        variance_of_gradient = float(gradient_magnitude.var())

        # Laplacian フィルタ
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=5)
        variance_of_laplacian = float(laplacian.var())

        # ガウシアンブラー適用後の差分
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        difference = cv2.absdiff(gray_image, blurred)
        variance_of_difference = float(difference.var())

        # 統合 raw スコア（大きいほどシャープ）
        blurriness_raw = (
            self.grad_weight * variance_of_gradient
            + self.lap_weight * variance_of_laplacian
            + self.diff_weight * variance_of_difference
        )

        # 5段階スコア＋ラベルに変換
        blurriness_score, blurriness_grade = self._to_score_and_grade(blurriness_raw)

        result = {
            "blurriness_score": blurriness_score,  # 0, 0.25, 0.5, 0.75, 1.0
            "blurriness_grade": blurriness_grade,  # very_blurry / blurry / fair / good / excellent
            "blurriness_raw": blurriness_raw,      # 連続値（チューニング/解析用）
            "variance_of_gradient": variance_of_gradient,
            "variance_of_laplacian": variance_of_laplacian,
            "variance_of_difference": variance_of_difference,
            "success": True,
        }

        return result
