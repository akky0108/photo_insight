import cv2
import rawpy
from skimage.metrics import structural_similarity as ssim
from file_handler.exif_file_handler import ExifFileHandler
from image_loader import ImageLoader
from utils.app_logger import Logger


class CompositionEvaluator:
    def __init__(self, image_path=None, logger=None, weights=None):
        self.loader = ImageLoader(logger)
        self.exif_handler = ExifFileHandler()
        self.logger = (
            logger
            if logger
            else Logger(logger_name="CompositionEvaluator").get_logger()
        )

        self.weights = (
            weights
            if weights
            else {"thirds": 1.0, "golden": 0.8, "symmetry": 1.0, "depth": 1.0}
        )

        if image_path:
            self.load_image(image_path)

    def load_image(self, image_path):
        try:
            if image_path.lower().endswith(".nef"):
                self.image = self.load_raw_image(image_path)
            else:
                self.image = self.loader.load_image(image_path)
            self.image = self.correct_rotation(image_path)
            self.height, self.width, _ = self.image.shape
        except Exception as e:
            self.logger.error(f"画像の読み込みに失敗しました ({image_path}): {e}")
            self.image = None

    def load_raw_image(self, image_path):
        try:
            with rawpy.imread(image_path) as raw:
                rgb_image = raw.postprocess()
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"RAW画像の処理に失敗しました ({image_path}): {e}")
            return None

    def correct_rotation(self, image_path):
        exif_data = self.exif_handler.get_exif_data(image_path)
        orientation = exif_data.get("Orientation", 1)
        if self.image is None:
            return None

        if orientation == 3:
            return cv2.rotate(self.image, cv2.ROTATE_180)
        elif orientation == 6:
            return cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            return cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return self.image

    def evaluate_rule_of_thirds(self):
        if self.image is None:
            return None
        # 中心座標ではなく、明るさの重心を使用
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        if moments["m00"] == 0:
            return 0
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        thirds_x = [self.width / 3, 2 * self.width / 3]
        thirds_y = [self.height / 3, 2 * self.height / 3]

        distance_to_thirds = min(
            min(abs(centroid_x - thirds_x[0]), abs(centroid_x - thirds_x[1])),
            min(abs(centroid_y - thirds_y[0]), abs(centroid_y - thirds_y[1])),
        )
        return max(0, 1 - (distance_to_thirds / (min(self.width, self.height) / 6)))

    def evaluate_symmetry(self):
        if self.image is None:
            return None
        left_half = self.image[:, : self.width // 2]
        right_half = cv2.flip(self.image[:, self.width // 2 :], 1)
        right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
        gray_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
        score_symmetry = ssim(gray_left, gray_right)
        return score_symmetry

    def evaluate_depth_and_focus(self):
        if self.image is None:
            return None
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian / 500, 1.0)

    def evaluate_composition(self):
        if self.image is None:
            return None
        thirds_score = self.evaluate_rule_of_thirds() or 0
        symmetry_score = self.evaluate_symmetry() or 0
        depth_score = self.evaluate_depth_and_focus() or 0
        weighted_score = (
            thirds_score * self.weights["thirds"]
            + symmetry_score * self.weights["symmetry"]
            + depth_score * self.weights["depth"]
        )
        normalized_score = weighted_score / sum(self.weights.values())
        return round(normalized_score * 100, 2)

    def evaluate_batch(self, image_paths):
        scores = {}
        for image_path in image_paths:
            try:
                self.load_image(image_path)
                scores[image_path] = self.evaluate_composition()
            except Exception as e:
                self.logger.error(f"{image_path} の評価中にエラー: {e}")
                scores[image_path] = None
        return scores


# 使用例
if __name__ == "__main__":
    image_paths = ["portrait1.nef", "portrait2.nef", "portrait3.nef"]
    weights = {"thirds": 1.0, "golden": 0.8, "symmetry": 1.0, "depth": 1.2}
    evaluator = CompositionEvaluator(weights=weights)
    batch_scores = evaluator.evaluate_batch(image_paths)
    for image_path, score in batch_scores.items():
        if score is not None:
            print(f"{image_path} - 構図スコア: {score:.2f}")
        else:
            print(f"{image_path} - 評価に失敗しました")
