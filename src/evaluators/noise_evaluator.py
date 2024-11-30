import cv2
import numpy as np
from utils.image_utils import ImageUtils

class NoiseEvaluator:
    """
    画像のノイズを評価するクラス。
    """
    def evaluate(self, image: np.ndarray) -> dict:
        """
        画像のノイズを評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: ノイズ評価の結果（スコアと成功フラグ）
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid input: expected a numpy array representing an image.")
        
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

        # スコアを 0-100 の範囲に正規化（max_noise_value は調整可能）
        normalized_score = (noise_std / 255.0) * 100

        # 結果を辞書で返す
        result = {
            'noise_score': normalized_score,  # 正規化されたノイズスコア
            'success': normalized_score > 0,   # スコアが0より大きいかどうかのフラグ
        }

        return result
