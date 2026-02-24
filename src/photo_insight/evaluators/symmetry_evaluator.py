import cv2
import numpy as np


class SymmetryEvaluator:
    """
    画像の対称性を評価するクラス。

    対称性は、左右または上下がほぼ同じ形状を持つことを意味し、バランスの取れた構図を生み出す要素です。
    このクラスでは、左右対称性を中心に画像の対称性を評価します。
    """

    def __init__(self, image):
        """
        コンストラクタ。評価する画像を受け取る。

        :param image: 評価対象の画像 (numpy配列)。
        """
        self.image = image
        self.height, self.width = self.image.shape[:2]

    def evaluate(self):
        """
        画像の左右対称性を評価する。

        左半分と右半分を比較し、どれだけ一致しているかを基にスコアを計算します。

        :return: スコア (0.0 ~ 1.0)。1.0は完全な対称性を表します。
        """
        # 画像の左半分と右半分を分割
        left_half = self.image[:, : self.width // 2]
        right_half = self.image[:, self.width // 2 :]

        # 右半分を左右反転させる
        right_half_flipped = cv2.flip(right_half, 1)

        # 左右のサイズが異なる場合、右半分を左半分のサイズにリサイズ
        if left_half.shape != right_half_flipped.shape:
            right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))

        # 左右のピクセル差を計算し、対称性をスコアリング
        # ピクセル差が少ないほどスコアが高くなる
        score_symmetry = 1 - np.mean(np.abs(left_half.astype("float") - right_half_flipped.astype("float")) / 255)

        return score_symmetry
