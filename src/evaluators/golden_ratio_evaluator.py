import cv2


class GoldenRatioEvaluator:
    """
    黄金比に基づいて画像構図を評価するクラス。

    黄金比は約1:1.618の比率で、古くから美しいとされる比率です。
    この比率を基に、重要な被写体が適切な位置に配置されているかを評価します。
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
        黄金比に基づいて画像構図を評価する。

        画像の中心点が黄金比に基づくラインや交差点にどれだけ近いかを基にスコアを計算します。

        :return: スコア (0.0 ~ 1.0)。1.0は理想的な構図を表します。
        """
        # 黄金比 (約0.618) に基づいて画像の縦横にラインを引く位置を計算
        golden_ratio = 0.618
        golden_x = [golden_ratio * self.width, (1 - golden_ratio) * self.width]
        golden_y = [golden_ratio * self.height, (1 - golden_ratio) * self.height]

        # 画像の中心点
        center_point = (self.width / 2, self.height / 2)

        # 画像の中心点から最も近い黄金比ラインまでの距離を計算
        distance_to_golden = min(
            min(abs(center_point[0] - golden_x[0]), abs(center_point[0] - golden_x[1])),
            min(abs(center_point[1] - golden_y[0]), abs(center_point[1] - golden_y[1])),
        )

        # 黄金比ラインに近いほどスコアが高くなるように計算
        # 最小値（画像の幅や高さの6分の1）に基づいてスコアを正規化
        score_golden = max(
            0, 1 - (distance_to_golden / (min(self.width, self.height) / 6))
        )

        return score_golden
