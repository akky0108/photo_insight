from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_face_evaluator import BaseFaceDetector

logger = logging.getLogger(__name__)


class InsightFaceDetector(BaseFaceDetector):
    """
    InsightFace を使用した顔検出クラス。
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        gpu: bool = False,
        *,
        strict: bool = False,
        model_name: str = "buffalo_l",
        model_root: str = "/work/models/insightface",
        providers: Optional[List[str]] = None,
        det_size: Tuple[int, int] = (640, 640),
        enable_eye_closed_estimation: bool = True,
    ):
        super().__init__(confidence_threshold)

        self._strict = bool(strict)
        self._model_name = str(model_name)
        self._model_root = str(model_root)
        self._gpu = bool(gpu)
        self._enable_eye_closed_estimation = bool(enable_eye_closed_estimation)

        if len(det_size) != 2:
            raise ValueError("det_size must be a tuple of (width, height).")
        self._det_size = (int(det_size[0]), int(det_size[1]))

        if providers:
            self._providers = list(providers)
        else:
            self._providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"] if self._gpu else ["CPUExecutionProvider"]
            )

        self._ctx_id = 0 if self._gpu else -1

        self.app = None
        self._init_error: Optional[str] = None

        try:
            self.setup()
        except Exception as e:
            self._init_error = str(e)
            self.app = None
            if self._strict:
                raise RuntimeError("InsightFaceDetector init failed. " f"detail={e}") from e

            logger.warning(
                "InsightFaceDetector initialization failed; fallback to empty detection. detail=%s",
                e,
                exc_info=True,
            )

    def available(self) -> bool:
        return self.app is not None

    def setup(self) -> None:
        if self.app is not None:
            return
        self.app = self._initialize_app()

    def _lazy_import_faceanalysis(self):
        try:
            import onnxruntime  # noqa: F401
        except Exception as e:
            raise ModuleNotFoundError("onnxruntime is not installed") from e

        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as e:
            raise ModuleNotFoundError("insightface is not installed") from e

        return FaceAnalysis

    def _lazy_import_cv2(self):
        try:
            import cv2  # type: ignore

            return cv2
        except Exception:
            return None

    def _initialize_app(self):
        FaceAnalysis = self._lazy_import_faceanalysis()

        app = FaceAnalysis(
            name=self._model_name,
            root=self._model_root,
            providers=self._providers,
        )
        app.prepare(
            ctx_id=self._ctx_id,
            det_size=self._det_size,
        )
        return app

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        if self.app is None:
            return {"faces": [], "face_detected": False, "num_faces": 0}

        gray_image: Optional[np.ndarray] = None
        if self._enable_eye_closed_estimation:
            gray_image = self._to_gray(image)

        try:
            faces_raw = self.app.get(image)
        except Exception as e:
            logger.warning(
                "InsightFace detect failed. detail=%s",
                e,
                exc_info=True,
            )
            return {
                "faces": [],
                "face_detected": False,
                "num_faces": 0,
                "error": str(e),
            }

        results: List[Dict[str, Any]] = []
        for face in faces_raw:
            try:
                score = float(getattr(face, "det_score", 0.0))
                if score < self.confidence_threshold:
                    continue

                box = self._extract_box(face)
                landmarks = self._extract_landmarks(face)
                yaw, pitch, roll = self._extract_pose(face)
                gaze_vector = self._estimate_gaze_vector(yaw, pitch)

                if self._enable_eye_closed_estimation:
                    eye_lap_var, eye_closed_prob, eye_patch_size = self._estimate_eye_closed(
                        image=image,
                        gray_image=gray_image,
                        box=box,
                        landmarks=landmarks,
                    )
                else:
                    eye_lap_var, eye_closed_prob, eye_patch_size = 0.0, 0.0, 0

                results.append(
                    {
                        "box": box,
                        "confidence": score,
                        "landmarks": landmarks,
                        "yaw": float(yaw),
                        "pitch": float(pitch),
                        "roll": float(roll),
                        "gaze": gaze_vector,
                        "eye_lap_var": float(eye_lap_var),
                        "eye_closed_prob": float(eye_closed_prob),
                        "eye_patch_size": int(eye_patch_size),
                    }
                )
            except Exception:
                continue

        return {
            "faces": results,
            "face_detected": bool(results),
            "num_faces": len(results),
        }

    def _extract_box(self, face) -> List[int]:
        return face.bbox.astype(int).tolist()

    def _extract_landmarks(self, face) -> Dict[str, Tuple[int, int]]:
        landmarks = face.kps.astype(int).tolist()
        return {
            "left_eye": (int(landmarks[0][0]), int(landmarks[0][1])),
            "right_eye": (int(landmarks[1][0]), int(landmarks[1][1])),
            "nose": (int(landmarks[2][0]), int(landmarks[2][1])),
            "mouth_left": (int(landmarks[3][0]), int(landmarks[3][1])),
            "mouth_right": (int(landmarks[4][0]), int(landmarks[4][1])),
        }

    def _extract_pose(self, face) -> Tuple[float, float, float]:
        pose = getattr(face, "pose", None)
        if pose is not None:
            return float(pose[0]), float(pose[1]), float(pose[2])

        pose2 = getattr(face, "normed_pose", None)
        if pose2 is not None:
            return float(pose2[0]), float(pose2[1]), float(pose2[2])

        return 0.0, 0.0, 0.0

    def _estimate_gaze_vector(self, yaw: float, pitch: float) -> Dict[str, float]:
        yaw_rad = np.radians(float(yaw))
        pitch_rad = np.radians(float(pitch))

        return {
            "x": float(np.sin(yaw_rad)),
            "y": float(np.sin(pitch_rad)),
            "z": float(np.cos(yaw_rad) * np.cos(pitch_rad)),
        }

    def _to_gray(self, image: np.ndarray) -> Optional[np.ndarray]:
        cv2 = self._lazy_import_cv2()
        if cv2 is None:
            return None

        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

    def _estimate_eye_closed(
        self,
        *,
        image: np.ndarray,
        gray_image: Optional[np.ndarray],
        box: List[int],
        landmarks: Dict[str, Tuple[int, int]],
        patch_scale: float = 0.18,
        t_closed: float = 60.0,
        t_open: float = 200.0,
    ) -> Tuple[float, float, int]:
        try:
            cv2 = self._lazy_import_cv2()
            if cv2 is None:
                return 0.0, 0.0, 0

            lx, ly = landmarks.get("left_eye", (None, None))
            rx, ry = landmarks.get("right_eye", (None, None))
            if lx is None or rx is None:
                return 0.0, 0.0, 0

            x1, _, x2, _ = box
            bw = max(1, int(x2 - x1))
            ps = max(12, int(bw * float(patch_scale)))

            def _crop(src: np.ndarray, cx: int, cy: int) -> np.ndarray:
                xx1 = max(0, cx - ps)
                yy1 = max(0, cy - ps)
                xx2 = min(int(src.shape[1]), cx + ps)
                yy2 = min(int(src.shape[0]), cy + ps)
                return src[yy1:yy2, xx1:xx2]

            src = gray_image if gray_image is not None else self._to_gray(image)
            if src is None:
                return 0.0, 0.0, 0

            patch_l = _crop(src, int(lx), int(ly))
            patch_r = _crop(src, int(rx), int(ry))

            def _lap_var(patch: np.ndarray) -> float:
                if patch.size == 0:
                    return 0.0
                return float(cv2.Laplacian(patch, cv2.CV_64F).var())

            v = (_lap_var(patch_l) + _lap_var(patch_r)) / 2.0

            if t_open <= t_closed:
                t_open = t_closed + 1.0

            open_prob = (v - t_closed) / (t_open - t_closed)
            open_prob = float(np.clip(open_prob, 0.0, 1.0))
            return float(v), float(1.0 - open_prob), int(ps)

        except Exception:
            return 0.0, 0.0, 0
