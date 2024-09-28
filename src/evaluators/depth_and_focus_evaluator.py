import cv2
import numpy as np

class DepthAndFocusEvaluator:
    """
    画像の焦点と深度を評価するクラス。
    
    焦点と深度は、画像内のどの部分が鮮明か、またはぼけているかを示し、視覚的な注目ポイントや
    構図の奥行きを決定する重要な要素です。
    このクラスでは、エッジ検出を使用して焦点の評価を行います。
    """
    def __init__(self, image):
        """
        コンストラクタ。評価する画像を受け取る。

        :param image: 評価対象の画像 (numpy配列)。
        """
        self.image = image

    def evaluate(self):
        """
        エッジ検出を使用して画像の焦点と深度を評価する。

        鮮明さと奥行きに基づきスコアを計算します。

        :return: スコア (0.0 ~ 1.0)。1.0は焦点と深度が非常に良好であることを表します。
        """
        # グレースケール画像に変換
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # エッジを検出 (Cannyエッジ検出)
        edges = cv2.Canny(gray_image, 100, 200)

        # エッジの強度に基づいてスコアを計算
        # エッジが多いほど鮮明な画像とみなし、スコアを高くする
        edge_density = np.mean(edges) / 255  # エッジピクセルの平均値を正規化
        focus_score = min(1.0, edge_density * 2)  # エッジ密度に基づきスコアを調整

        return focus_score
