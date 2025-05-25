import numpy as np
from typing import List, Optional, Dict, Any

from evaluators.face_evaluator import FaceEvaluator
from utils.app_logger import Logger

class FaceProcessor:
    def __init__(self, face_evaluator: FaceEvaluator, logger: Optional[Logger] = None):
        self.face_evaluator = face_evaluator
        self.logger = logger or Logger("FaceProcessor")

    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            result = self.face_evaluator.evaluate(image)
            self.logger.info(f"顔検出結果: {result}")
            return result
        except Exception as e:
            self.logger.warning(f"顔検出中にエラー: {str(e)}")
            return {"face_score": 0, "faces": []}

    def get_best_face(self, faces: list) -> Optional[dict]:
        if not faces:
            return None
        return max(faces, key=lambda f: f.get("confidence", 0))

    def crop_face(self, image: np.ndarray, face: dict) -> Optional[np.ndarray]:
        box = face.get("box") or face.get("bbox")
        if box and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            return image[y1:y2, x1:x2]
        return None

    def extract_attributes(self, face: dict) -> Dict[str, Any]:
        return {attr: face[attr] for attr in ["yaw", "pitch", "roll", "gaze"] if attr in face}

