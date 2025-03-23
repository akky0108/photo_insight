import cv2
import numpy as np
from utils.image_utils import ImageUtils

class BlurrinessEvaluator:
    """
    ぼやけ具合を評価するクラス（複数のアルゴリズムを使用）
    """

    def evaluate(self, image: np.ndarray) -> dict:
        """
        画像のぼやけ具合を評価する。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: ブレのスコアを含む辞書
        """
        # グレースケール変換
        if len(image.shape) == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # Sobelエッジ検出（Tenengrad法）
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 勾配のスケーリング（最大値を制限）
        gradient_magnitude = np.clip(gradient_magnitude, 0, 500)  # 最大値を500に制限

        # 勾配の分散（ぼやけ具合を示す指標）
        variance_of_gradient = gradient_magnitude.var()

        # Laplacianフィルタ（フォーカスマップ用）
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=5)
        variance_of_laplacian = laplacian.var()

        # ガウシアンブラー適用後の差分（ぼけの度合いを強調）
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        difference = cv2.absdiff(gray_image, blurred)
        variance_of_difference = difference.var()

        # 統合スコア（重みを調整）
        blurriness_score = (
            0.4 * variance_of_gradient +
            0.4 * variance_of_laplacian +
            0.2 * variance_of_difference
        )

        # スコアの正規化（最大値と最小値を使ってスケーリング）
        min_score = min(variance_of_gradient, variance_of_laplacian, variance_of_difference)
        max_score = max(variance_of_gradient, variance_of_laplacian, variance_of_difference)

        # スコアの範囲を0〜1にスケーリング
        if max_score > min_score:
            blurriness_score = (blurriness_score - min_score) / (max_score - min_score)
        else:
            blurriness_score = 0  # すべての指標が同じ値であればスコアは0とみなす

        # スコアの最大値を制限（極端に大きなスコアを制限）
        blurriness_score = min(blurriness_score, 1.0)  # 最大値を1.0に制限

        # 結果を辞書形式で返す
        result = {
            'blurriness_score': blurriness_score,
            'variance_of_gradient': variance_of_gradient,
            'variance_of_laplacian': variance_of_laplacian,
            'variance_of_difference': variance_of_difference,
            'success': True
        }

        return result
