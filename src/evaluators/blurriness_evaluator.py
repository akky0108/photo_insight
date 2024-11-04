import cv2
import numpy as np
from utils.image_utils import ImageUtils

# Blurriness Evaluator with Tenengrad Method
class BlurrinessEvaluator:
    """
    Tenengrad法を用いて画像のぼやけ具合を評価するクラス。
    """

    def evaluate(self, image: np.ndarray) -> dict:
        """
        Tenengrad法を用いて画像のぼやけ具合を評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: ブレのスコアを含む辞書
        """
        # 画像がカラーの場合はグレースケールに変換
        if len(image.shape) == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # Sobelフィルタを適用してエッジを検出
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # 勾配の大きさを計算
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 勾配の大きさの分散を計算（ぼやけ具合を示す指標）
        variance_of_gradient = gradient_magnitude.var()

        # 結果を辞書形式で返す
        result = {
            'blurriness_score': variance_of_gradient,  # ぼやけ具合のスコア
            'success': True if variance_of_gradient is not None else False  # 成功フラグ
        }

        return result
