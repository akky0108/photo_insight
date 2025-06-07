from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import logging
from typing import Optional


class FaceDetectorBase(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        顔を検出し、結果を返すメソッド
        :param image: 入力画像（numpy配列）
        :return: 顔検出結果（辞書形式）
            'faces' (list): 検出された顔の情報のリスト
                - 'box' (list[int]): 顔のバウンディングボックス [x, y, width, height]
                - 'confidence' (float): 顔検出の信頼度
                - 'landmarks' (dict): 顔のランドマーク位置（例: 'left_eye', 'right_eye' 等）
            'face_detected' (bool): 顔が検出されたか
            'num_faces' (int): 検出された顔の数
        """
        pass


class BaseFaceDetector(FaceDetectorBase):
    def __init__(
        self, confidence_threshold=0.5, logger: Optional[logging.Logger] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)

    def _check_image_size(self, image: np.ndarray) -> bool:
        """画像が適切なサイズかどうかを確認"""
        if image.shape[0] < 60 or image.shape[1] < 60:
            self.logger.error(f"Image size is too small: {image.shape}")
            return False
        return True

    def detect(self, image: np.ndarray) -> dict:
        try:
            if not self._check_image_size(image):
                raise ValueError("Image is too small for face detection")

            # 実際の顔検出処理はサブクラスで実装する
            faces = []  # ここでは仮実装として空のリストを返す

            return {
                "faces": faces,
                "face_detected": bool(faces),
                "num_faces": len(faces),
            }

        except Exception as e:
            self.logger.error(f"Error during face detection: {str(e)}")
            return {"error": str(e), "face_detected": False, "num_faces": 0}
