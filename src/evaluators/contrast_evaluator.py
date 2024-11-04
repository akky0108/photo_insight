import cv2
import numpy as np
from utils.image_utils import ImageUtils

class ContrastEvaluator:
    """
    画像のコントラストを評価するクラス。
    """
    def evaluate(self, image: np.ndarray) -> dict:
        """
        画像のコントラストを評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: コントラストの評価結果（スコアと成功フラグ）
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

        # 結果を辞書で返す
        result = {
            'contrast_score': normalized_score,  # 正規化されたコントラストスコア
            'success': normalized_score > 0,      # スコアが0より大きいかどうかのフラグ
        }

        return result
