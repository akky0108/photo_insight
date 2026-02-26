# src/photo_insight/face_detectors/face_processor.py
# -*- coding: utf-8 -*-
import numpy as np
from typing import Optional, Dict, Any, Sequence, Union

from photo_insight.evaluators.face_evaluator import FaceEvaluator
from photo_insight.core.logging import Logger

BoxLike = Union[Sequence[float], Sequence[int]]  # [x1, y1, x2, y2]


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

    # NOTE:
    # tests/unit/face_detectors/test_face_processor.py では
    # FaceProcessor.get_best_face(faces) のようにクラスから呼んでいるため staticmethod 化する。
    @staticmethod
    def get_best_face(faces: list) -> Optional[dict]:
        if not faces:
            return None
        return max(faces, key=lambda f: (f or {}).get("confidence", 0))

    # NOTE:
    # テストでは FaceProcessor.crop_face(image, box) のように bbox(list) を直接渡す。
    # 実装側（PortraitQualityEvaluator）では FaceProcessor.crop_face(image, face_dict) を渡す。
    # 両方を受けられるようにする。
    @staticmethod
    def crop_face(image: np.ndarray, face: Union[dict, BoxLike, None]) -> Optional[np.ndarray]:
        if image is None:
            return None

        box = None
        if isinstance(face, dict):
            box = face.get("box") or face.get("bbox")
        elif isinstance(face, (list, tuple)) and len(face) == 4:
            box = face

        if not box or len(box) != 4:
            return None

        try:
            x1, y1, x2, y2 = map(int, box)
        except Exception:
            return None

        h, w = image.shape[:2]
        # clamp to image bounds
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))

        # invalid or empty region
        if x2 <= x1 or y2 <= y1:
            return None

        return image[y1:y2, x1:x2]

    @staticmethod
    def extract_attributes(face: dict) -> Dict[str, Any]:
        if not isinstance(face, dict):
            return {}
        return {attr: face[attr] for attr in ["yaw", "pitch", "roll", "gaze"] if attr in face}
