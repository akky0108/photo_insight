import cv2
import numpy as np
from utils.image_utils import ImageUtils

class ContrastEvaluator:
    def evaluate(self, image: np.ndarray) -> float:
        """
        画像のコントラストを評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: コントラストのスコア（高いほどコントラストが強い）
        """
        # 画像がカラーの場合はグレースケールに変換
        if len(image.shape) == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # 画像の標準偏差を計算（標準偏差が大きいほどコントラストが高い）
        contrast_score = np.std(gray_image)

        return contrast_score
