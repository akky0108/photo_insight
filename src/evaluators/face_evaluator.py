import cv2
import numpy as np
import traceback
from log_util import Logger
from mtcnn import MTCNN

class FaceEvaluator:
    """
    MTCNNを用いて顔領域を検出するクラス。
    """

    def __init__(self):
        """
        初期化メソッド。
        """
        self.detector = MTCNN()
        self.logger = Logger()

    def evaluate(self, image):
        """
        画像内で顔を検出し、結果を返す。

        :param image: RGB形式の画像
        :return: 検出された顔領域のリストと成功フラグ
        """
        try:
            # MTCNNを用いて顔を検出
            detections = self.detector.detect_faces(image)

            # 検出された顔領域を保持するリスト
            faces = []

            # 信頼度が高い顔領域のみを抽出
            for detection in detections:
                confidence = detection['confidence']
                if confidence > 0.5:  # 信頼度が0.5以上の顔を採用
                    x, y, width, height = detection['box']
                    faces.append({
                        'box': (x, y, width, height),
                        'confidence': confidence
                    })

            # 検出された顔領域のリストと、検出が成功したかどうかを返す
            return faces, bool(faces)

        except Exception as e:
            # エラーハンドリング
            self.logger.error(f"Error during face detection: {e}")
            self.logger.error(traceback.format_exc())
            return {'error': 'Face detection failed'}, False
