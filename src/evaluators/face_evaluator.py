import cv2
import numpy as np
import traceback
from evaluators.sharpness_evaluator import SharpnessEvaluator
from evaluators.contrast_evaluator import ContrastEvaluator
from evaluators.noise_evaluator import NoiseEvaluator
from evaluators.wavelet_sharpness_evaluator import WaveletSharpnessEvaluator
from evaluators.blurriness_evaluator import BlurrinessEvaluator
from log_util import Logger  # ロギング用クラスをインポート
from utils.image_utils import ImageUtils  # 修正されたImageUtilsクラスをインポート

class FaceEvaluator:
    """
    顔領域を検出し、各種品質評価を行うクラス。
    """

    def __init__(self, weights=None):
        """
        初期化メソッド。デフォルトで評価項目ごとの重みを設定。

        :param weights: 各評価項目の重みを指定する辞書 (省略可能)
        """
        if weights is None:
            self.weights = {
                'sharpness': 0.2,
                'contrast': 0.2,
                'noise': 0.2,
                'wavelet_sharpness': 0.1,
                'blurriness': 0.1
            }
        else:
            self.weights = weights

        # 各種品質評価器を初期化
        self.sharpness_evaluator = SharpnessEvaluator()
        self.contrast_evaluator = ContrastEvaluator()
        self.noise_evaluator = NoiseEvaluator()
        self.wavelet_sharpness_evaluator = WaveletSharpnessEvaluator()
        self.blurriness_evaluator = BlurrinessEvaluator()

        # DNNベースの顔検出モデルをロード
        model_file = "/home/mluser/opencv_face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "/home/mluser/opencv_face_detection_model/deploy.prototxt"
        self.net = cv2.dnn.readNetFromCaffe(config_file, model_file)

        # ロガーを初期化
        self.logger = Logger()

    def detect_and_evaluate(self, image, is_raw=False):
        """
        画像内で顔を検出し、顔領域に対して品質評価を行う。

        :param image: RGB形式の画像またはRAW画像
        :param is_raw: RAW画像の場合True
        :return: 顔の品質評価結果、顔領域の重み、顔が検出されたかどうか
        """
        try:
            # RAW画像かどうかで前処理方法を切り替える
            if is_raw:
                blob = ImageUtils.preprocess_raw_image(image)
            else:
                blob = ImageUtils.preprocess_image(image)

            # DNNを用いた顔検出の実行
            self.net.setInput(blob)
            detections = self.net.forward()

            # 検出された顔領域を保持する変数
            faces = []

            # 検出結果から信頼度が高い顔領域を抽出
            h, w = image.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # 信頼度が50%以上のものを採用
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    faces.append((x, y, x1 - x, y1 - y))

            # 顔が検出されなかった場合、評価値を0として返す
            if len(faces) == 0:
                return {
                    'sharpness': 0,
                    'contrast': 0,
                    'noise_level': 0,
                    'wavelet_sharpness': 0,
                    'blurriness': 0
                }, 0, False

            # 最大の顔領域を選択し、顔領域を切り出す
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])  # 面積が最も大きい顔領域を選ぶ
            face_region = ImageUtils.extract_region(image, largest_face)  # 顔領域を切り出す

            # 必要に応じて顔領域をリサイズ（スケーリング）
            face_region_scaled = ImageUtils.scale_region(face_region)

            # 各種品質評価を実行
            face_evaluation = {
                'sharpness': self.sharpness_evaluator.evaluate(face_region_scaled),
                'contrast': self.contrast_evaluator.evaluate(face_region_scaled),
                'noise_level': self.noise_evaluator.evaluate(face_region_scaled),
                'wavelet_sharpness': self.wavelet_sharpness_evaluator.evaluate(face_region_scaled),
                'blurriness': self.blurriness_evaluator.evaluate(face_region_scaled)
            }

            # 顔領域の重みを画像全体に対する面積比で計算
            face_weight = ImageUtils.calculate_region_weight(largest_face, image.shape)

            return face_evaluation, face_weight, True

        except cv2.error as e:
            # OpenCVに関するエラーハンドリング
            self.logger.error(f"OpenCV error during face detection and evaluation: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'sharpness': 0,
                'contrast': 0,
                'noise_level': 0,
                'wavelet_sharpness': 0,
                'blurriness': 0
            }, 0, False

        except Exception as e:
            # その他の例外に対するエラーハンドリング
            self.logger.error(f"Unexpected error during face detection and evaluation: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'sharpness': 0,
                'contrast': 0,
                'noise_level': 0,
                'wavelet_sharpness': 0,
                'blurriness': 0
            }, 0, False
