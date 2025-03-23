import cv2
import numpy as np
import traceback
from mtcnn import MTCNN
from typing import Any, Dict, List, Optional, Tuple
import logging

class FaceEvaluator:
    """
    MTCNNを用いて顔領域とランドマークを検出するクラス。
    """

    def __init__(self, confidence_threshold: float = 0.5, logger: Optional[logging.Logger] = None):
        """
        初期化メソッド。

        :param confidence_threshold: 顔検出の信頼度閾値（デフォルト: 0.5）
        :param logger: ロガーオブジェクト（デフォルト: 標準出力にログを出力）
        """
        self.detector = MTCNN()
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)
        if not logger:
            logging.basicConfig(level=logging.INFO)

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像内で顔とランドマークを検出し、結果を返す。

        :param image: RGB形式の画像 (numpy配列)
        :return: 検出結果の辞書（顔領域、信頼度、ランドマーク、検出数、成功フラグなど）
        """
        try:
            # 入力画像がRGB形式かどうかをチェック
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise ValueError("Input image must be in RGB format with 3 channels.")

            # MTCNNを用いて顔を検出
            detections = self.detector.detect_faces(image)
            self.logger.info(f"Number of faces detected: {len(detections)}")

            # 検出された顔領域とランドマークを保持するリスト
            faces = []

            for detection in detections:
                confidence = detection.get('confidence', 0)
                if confidence >= self.confidence_threshold:
                    # 顔領域の座標
                    x, y, width, height = detection['box']
                    x, y = max(0, x), max(0, y)
                    width, height = min(image.shape[1] - x, width), min(image.shape[0] - y, height)

                    # ランドマーク情報（目、鼻、口の座標）
                    landmarks = detection.get('keypoints', {})

                    faces.append({
                        'box': (x, y, width, height),
                        'confidence': confidence,
                        'landmarks': landmarks  # 目、鼻、口の座標
                    })

            # 結果の辞書を作成して返す
            return {
                'faces': faces,          # 検出された顔のリスト
                'success': bool(faces),  # 検出成功フラグ
                'num_faces': len(faces), # 検出された顔の数
            }

        except Exception as e:
            # エラーハンドリング
            error_message = f"Error during face detection: {str(e)}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())
            return {
                'error': error_message,
                'success': False,
                'num_faces': 0
            }
