import cv2
import numpy as np
from utils.image_utils import ImageUtils

# Noise Evaluator
class NoiseEvaluator:
    def evaluate(self, image: np.ndarray) -> float:
        """
        画像のノイズレベルを評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: ノイズレベルのスコア（高いほどノイズが多い）
        """
        # 画像がカラーの場合はグレースケールに変換
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # ガウシアンブラーを適用してスムージング
        smoothed_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # 元の画像とスムージングした画像との差分を計算
        noise = gray_image - smoothed_image

        # ノイズの標準偏差を計算
        noise_std = np.std(noise)

        return noise_std
