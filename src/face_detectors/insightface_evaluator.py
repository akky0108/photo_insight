from insightface.app import FaceAnalysis
import numpy as np
from typing import Dict, Any
from base_face_evaluator import BaseFaceDetector


class InsightFaceDetector(BaseFaceDetector):
    def __init__(self, confidence_threshold: float = 0.5, gpu: bool = True):
        super().__init__(confidence_threshold)
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0 if gpu else -1)

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            faces_raw = self.app.get(image)
        except Exception as e:
            print(f"Error in face detection: {e}")
            return {'faces': [], 'face_detected': False, 'num_faces': 0}

        results = []
        for face in faces_raw:
            if face.det_score >= self.confidence_threshold:
                box = face.bbox.astype(int).tolist()
                landmarks = face.kps.astype(int).tolist()
                results.append({
                    'box': box,
                    'confidence': float(face.det_score),
                    'landmarks': {
                        'left_eye': landmarks[0],
                        'right_eye': landmarks[1],
                        'nose': landmarks[2],
                        'mouth_left': landmarks[3],
                        'mouth_right': landmarks[4]
                    }
                })
        return {
            'faces': results,
            'face_detected': bool(results),
            'num_faces': len(results)
        }