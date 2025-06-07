class RuleOfThirdsEvaluator:
    """
    ルール・オブ・サード（3分割法）に基づいて画像構図を評価するクラス。

    ルール・オブ・サードとは、画像を縦横3等分することで重要な要素を配置するポイントを決め、
    よりバランスの取れた構図を作るための指標です。
    画像の中心ではなく、3等分のライン上や交差点に主要な被写体を配置すると、より視覚的に
    良い構図になると言われています。
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
        ルール・オブ・サードに基づいて画像構図を評価する。

        画像の中心点が3分割のラインや交差点にどれだけ近いかを基にスコアを計算します。

        :return: スコア (0.0 ~ 1.0)。1.0は理想的な構図を表します。
        """
        # 画像の縦横それぞれを3分割したラインの位置を計算
        thirds_x = [self.width / 3, 2 * self.width / 3]
        thirds_y = [self.height / 3, 2 * self.height / 3]

        # 画像の中心点
        center_point = (self.width / 2, self.height / 2)

        # 画像の中心点から最も近いサードラインまでの距離を計算
        distance_to_thirds = min(
            min(abs(center_point[0] - thirds_x[0]), abs(center_point[0] - thirds_x[1])),
            min(abs(center_point[1] - thirds_y[0]), abs(center_point[1] - thirds_y[1])),
        )

        # サードラインに近いほどスコアが高くなるように計算
        # 最小値（画像の幅や高さの6分の1）に基づいてスコアを正規化
        score_thirds = max(
            0, 1 - (distance_to_thirds / (min(self.width, self.height) / 6))
        )

        return score_thirds
