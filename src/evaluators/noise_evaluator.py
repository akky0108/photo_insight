import cv2
import numpy as np

class NoiseEvaluator:
    """
    画像のノイズを評価するクラス。
    """
    def __init__(self, max_noise_value=70.0):
        """
        初期化メソッド。

        :param max_noise_value: ノイズの最大許容値（この値を基準にスコアを計算）
        """
        self.max_noise_value = max_noise_value

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

        # 逆スコア化: ノイズが少ないほど高いスコアにする
        normalized_score = max(0.0, 100 * (1 - (noise_std / self.max_noise_value)))
        normalized_score = np.clip(normalized_score, 0, 100)  # スコアを 0〜100 の範囲にクリップ

        # 結果を辞書で返す
        result = {
            'noise_score': normalized_score,  # 正規化されたノイズスコア
            'success': True if normalized_score > 0 else False,  # スコアが0以上なら成功とみなす
        }

        return result
