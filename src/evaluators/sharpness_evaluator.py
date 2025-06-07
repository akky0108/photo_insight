import cv2
import numpy as np
from utils.image_utils import ImageUtils


# Sharpness Evaluator
class SharpnessEvaluator:
    """
    シャープネスを評価するクラス。
    """

    def evaluate(self, image: np.ndarray) -> dict:
        """
        画像のシャープネスを評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: 評価結果を含む辞書
        """
        # 画像がカラーの場合はグレースケールに変換
        if len(image.shape) == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # ラプラシアンを計算
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        # ラプラシアンの分散を計算
        variance = laplacian.var()

        result = {
            "sharpness_score": variance,  # シャープネススコア（高いほどシャープ）
            "success": True,  # 成功フラグ（常に成功と仮定）
        }

        return result
