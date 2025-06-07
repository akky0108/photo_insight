import cv2
import numpy as np


class LocalContrastEvaluator:
    def __init__(self, block_size: int = 32):
        """
        :param block_size: 局所コントラストを測るブロックのサイズ
        """
        self.block_size = block_size

    def evaluate(self, image: np.ndarray) -> dict:
        """
        画像の局所コントラストを評価する。

        :param image: 評価対象の画像 (numpy.ndarray)
        :return: 局所コントラストの平均・標準偏差
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        contrast_values = []

        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                block = gray[y : y + self.block_size, x : x + self.block_size]
                if block.shape[0] < self.block_size or block.shape[1] < self.block_size:
                    continue

                block_max = float(block.max())
                block_min = float(block.min())
                contrast = (block_max - block_min) / (block_max + block_min + 1e-5)
                contrast_values.append(contrast)

        if not contrast_values:
            return {"local_contrast_score": 0.0, "local_contrast_std": 0.0}

        return {
            "local_contrast_score": np.mean(contrast_values),
            "local_contrast_std": np.std(contrast_values),  # コントラストのばらつき
        }