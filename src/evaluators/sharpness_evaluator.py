import cv2
import numpy as np
from utils.image_utils import ImageUtils

# Sharpness Evaluator
class SharpnessEvaluator:
    def evaluate(self, image: np.ndarray) -> float:
        """
        画像のシャープネスを評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: シャープネススコア（高いほどシャープ）
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

        return variance