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
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid input: expected a numpy array representing an image.")
        
        # 画像がカラーの場合はグレースケールに変換
        if len(image.shape) == 3 and image.shape[2] == 3:  # BGR チェック
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # 画像の標準偏差を計算（標準偏差が大きいほどコントラストが高い）
        contrast_score = np.std(gray_image)
        
        # スコアを 0-100 の範囲に正規化する (max_contrast_value は調整可能)
        normalized_score = (contrast_score / 255.0) * 100

        return normalized_score
