import cv2
import numpy as np
class LocalSharpnessEvaluator:
    def __init__(self, block_size: int = 32):
        """
        :param block_size: 局所シャープネスを測るブロックのサイズ
        """
        self.block_size = block_size

    def evaluate(self, image: np.ndarray) -> dict:
        """
        画像の局所シャープネスを評価する。

        :param image: 評価対象の画像 (numpy.ndarray)
        :return: 局所シャープネスの平均・標準偏差
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        sharpness_values = []

        # 画像をブロック単位で処理
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                block = gray[y : y + self.block_size, x : x + self.block_size]
                if block.shape[0] < self.block_size or block.shape[1] < self.block_size:
                    continue
                laplacian = cv2.Laplacian(block, cv2.CV_64F)
                sharpness_values.append(np.var(laplacian))

        if not sharpness_values:
            return {"local_sharpness_score": 0.0, "local_sharpness_std": 0.0}

        # 局所シャープネスの平均値と標準偏差を返す
        return {
            "local_sharpness_score": np.mean(sharpness_values),
            "local_sharpness_std": np.std(
                sharpness_values
            ),  # 局所的なシャープネスのばらつき
        }
