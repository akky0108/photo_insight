import numpy as np
from typing import Dict, Any

from face_detectors.insightface_evaluator import InsightFaceDetector
from face_detectors.mtcnn_face_evaluator import MtcnnFaceDetector


class FaceEvaluator:
    def __init__(self, backend: str = 'mtcnn', confidence_threshold: float = 0.5):
        if backend == 'insightface':
            self.detector = InsightFaceDetector(confidence_threshold=confidence_threshold)
        elif backend == 'mtcnn':
            self.detector = MtcnnFaceDetector(confidence_threshold=confidence_threshold)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        return self.detector.detect(image)
