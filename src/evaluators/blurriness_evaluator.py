import cv2
import numpy as np
from utils.image_utils import ImageUtils

# Blurriness Evaluator
class BlurrinessEvaluator:
    def evaluate(self, image: np.ndarray) -> float:
        """
        画像のぼやけ具合を評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: ブレのスコア（低いほどぼやけている）
        """
        # 画像がカラーの場合はグレースケールに変換
        if len(image.shape) == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # ラプラシアンフィルタを適用してエッジ強度を計算
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        # エッジの分散（エッジのシャープさを示す指標）を計算
        variance_of_laplacian = laplacian.var()

        return variance_of_laplacian
