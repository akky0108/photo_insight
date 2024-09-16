import cv2
import numpy as np
from file_handler.exif_file_handler import ExifFileHandler
from image_loader import ImageLoader
from log_util import Logger


class CompositionEvaluator:
    """
    ルール・オブ・サード、黄金比、対称性、焦点と深度などの規則に基づき、画像構図を評価するクラス。
    """
    def __init__(self, image_path=None, logger=None, weights=None):
        """
        コンストラクタ。画像パス、ロガー、評価の重みをオプションで受け取る。

        :param image_path: 読み込む画像ファイルのパス。
        :param logger: カスタムロガーのインスタンス。
        :param weights: 各構図ルールに対する重み。
        """
        self.loader = ImageLoader(logger)
        self.exif_handler = ExifFileHandler()
        self.logger = logger if logger else Logger(logger_name='CompositionEvaluator').get_logger()

        # 重みが指定されていない場合はデフォルトの重みを設定
        self.weights = weights if weights else {
            'thirds': 1.0,
            'golden': 0.8,
            'symmetry': 1.0,
            'depth': 1.0
        }

        # 画像パスが指定されている場合は画像を読み込む
        if image_path:
            self.load_image(image_path)

    def load_image(self, image_path):
        """
        画像を読み込み、EXIFデータに基づいて回転を補正する。
        """
        self.image = self.loader.load_image(image_path)
        self.image = self.correct_rotation(image_path)
        self.height, self.width, _ = self.image.shape

    def correct_rotation(self, image_path):
        """
        EXIFデータを使用して画像の回転を補正する。

        :param image_path: 補正する画像ファイルのパス。
        :return: 回転された画像。
        """
        exif_data = self.exif_handler.get_exif_data(image_path)
        orientation = exif_data.get('Orientation', 1)

        if orientation == 3:
            self.image = cv2.rotate(self.image, cv2.ROTATE_180)
        elif orientation == 6:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return self.image

    def evaluate_rule_of_thirds(self):
        """
        ルール・オブ・サードに基づいて画像構図を評価する。
        """
        thirds_x = [self.width / 3, 2 * self.width / 3]
        thirds_y = [self.height / 3, 2 * self.height / 3]
        center_point = (self.width / 2, self.height / 2)

        distance_to_thirds = min(
            min(abs(center_point[0] - thirds_x[0]), abs(center_point[0] - thirds_x[1])),
            min(abs(center_point[1] - thirds_y[0]), abs(center_point[1] - thirds_y[1]))
        )

        score_thirds = max(0, 1 - (distance_to_thirds / (min(self.width, self.height) / 6)))
        return score_thirds

    def evaluate_golden_ratio(self):
        """
        黄金比に基づいて画像構図を評価する。
        """
        golden_ratio = 0.618
        golden_x = [golden_ratio * self.width, (1 - golden_ratio) * self.width]
        golden_y = [golden_ratio * self.height, (1 - golden_ratio) * self.height]
        center_point = (self.width / 2, self.height / 2)

        distance_to_golden = min(
            min(abs(center_point[0] - golden_x[0]), abs(center_point[0] - golden_x[1])),
            min(abs(center_point[1] - golden_y[0]), abs(center_point[1] - golden_y[1]))
        )

        score_golden = max(0, 1 - (distance_to_golden / (min(self.width, self.height) / 6)))
        return score_golden

    def evaluate_symmetry(self):
        """
        画像の対称性を評価する。
        """
        left_half = self.image[:, :self.width // 2]
        right_half = cv2.flip(self.image[:, self.width // 2:], 1)

        # 比較のために両方の半分のサイズを同一にする
        if left_half.shape != right_half.shape:
            right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))

        score_symmetry = 1 - np.mean(np.abs(left_half.astype("float") - right_half.astype("float")) / 255)
        return score_symmetry

    def evaluate_depth_and_focus(self):
        """
        エッジ検出を使用して、画像の焦点と深度を評価する。
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)

        # 背景と前景のエッジの違いを測定する
        depth_edges = cv2.GaussianBlur(edges, (5, 5), 0)
        score_depth = 1 - np.mean(np.abs(gray_image.astype("float") - depth_edges.astype("float")) / 255)

        return score_depth

    def evaluate_composition(self):
        """
        各ルールの評価を組み合わせ、全体の構図を評価する。
        """
        thirds_score = self.evaluate_rule_of_thirds()
        golden_score = self.evaluate_golden_ratio()
        symmetry_score = self.evaluate_symmetry()
        depth_score = self.evaluate_depth_and_focus()

        weighted_score = (
            thirds_score * self.weights['thirds'] +
            golden_score * self.weights['golden'] +
            symmetry_score * self.weights['symmetry'] +
            depth_score * self.weights['depth']
        )

        normalized_score = weighted_score / sum(self.weights.values())
        final_score = normalized_score * 100
        return round(float(final_score), 2)

    def evaluate_batch(self, image_paths):
        """
        複数の画像を一括して評価し、それぞれの構図スコアを返す。

        :param image_paths: 画像ファイルのパスのリスト。
        :return: 画像パスをキーにして構図スコアを値に持つ辞書。
        """
        scores = {}
        for image_path in image_paths:
            try:
                self.load_image(image_path)
                scores[image_path] = self.evaluate_composition()
            except Exception as e:
                self.logger.error(f"{image_path} の評価中にエラーが発生しました: {e}")
                scores[image_path] = None
        return scores


# 使用例
if __name__ == "__main__":
    image_paths = ["portrait1.nef", "portrait2.nef", "portrait3.nef"]
    weights = {
        'thirds': 1.0,
        'golden': 0.8,
        'symmetry': 1.0,
        'depth': 1.2
    }
    evaluator = CompositionEvaluator(weights=weights)
    batch_scores = evaluator.evaluate_batch(image_paths)
    for image_path, score in batch_scores.items():
        if score is not None:
            print(f"{image_path} - 構図スコア: {score:.2f}")
        else:
            print(f"{image_path} - 評価に失敗しました")
