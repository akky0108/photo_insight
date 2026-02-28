# src/photo_insight/face_detectors/insightface_evaluator.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

from .base_face_evaluator import BaseFaceDetector


class InsightFaceDetector(BaseFaceDetector):
    """
    InsightFace を使用した顔検出クラス。

    ✅ 方針（CI耐性）:
    - insightface が無い環境でも「import しただけ」で落ちない
    - insightface が無い場合:
        - strict=False(デフォルト): detect() は空結果を返す（CI向け）
        - strict=True: __init__ で RuntimeError（運用向け）

    - バウンディングボックス、ランドマーク、姿勢情報（yaw, pitch, roll）を取得可能。
    - OCPを意識し、処理ロジックをメソッド単位で分離。
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        gpu: bool = True,
        *,
        strict: bool = False,
        model_name: str = "buffalo_l",
    ):
        """
        :param confidence_threshold: 顔検出の信頼度しきい値
        :param gpu: True: GPU使用, False: CPU使用
        :param strict: True の場合、insightface が無ければ初期化で例外
        :param model_name: FaceAnalysis のモデル名
        """
        super().__init__(confidence_threshold)
        self._strict = bool(strict)
        self._model_name = str(model_name)
        self._gpu = bool(gpu)

        self.app = None
        self._init_error: Optional[str] = None

        try:
            FaceAnalysis = self._lazy_import_faceanalysis()
            self.app = FaceAnalysis(name=self._model_name)
            self.app.prepare(ctx_id=0 if self._gpu else -1)
        except Exception as e:
            self._init_error = str(e)
            self.app = None
            if self._strict:
                raise RuntimeError(
                    "InsightFaceDetector init failed. "
                    "insightface が未導入、またはモデル初期化に失敗しました。"
                    f" detail={e}"
                )

    def _lazy_import_faceanalysis(self):
        """
        insightface を遅延 import。
        """
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except Exception as e:
            raise ModuleNotFoundError("insightface is not installed (required for InsightFaceDetector).") from e
        return FaceAnalysis

    def available(self) -> bool:
        """
        実行環境で insightface detector が利用可能かどうか。
        """
        return self.app is not None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        顔検出を行い、検出された顔ごとに情報を抽出する。

        :param image: BGR形式のNumPy配列画像
        :return: 顔検出結果 dict
        """
        if self.app is None:
            # CI / lightweight env: return empty safely
            return {"faces": [], "face_detected": False, "num_faces": 0}

        try:
            faces_raw = self.app.get(image)
        except Exception as e:
            # ここで print はテストログ汚しやすいので dict に寄せる
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

                eye_lap_var, eye_closed_prob, eye_patch_size = self._estimate_eye_closed(
                    image=image,
                    box=box,
                    landmarks=landmarks,
                )

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
                # 1 face の失敗で全体を落とさない（頑健性）
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

        x = float(np.sin(yaw_rad))
        y = float(np.sin(pitch_rad))
        z = float(np.cos(yaw_rad) * np.cos(pitch_rad))

        return {"x": x, "y": y, "z": z}

    def _estimate_eye_closed(
        self,
        *,
        image: np.ndarray,
        box: List[int],
        landmarks: Dict[str, Tuple[int, int]],
        patch_scale: float = 0.18,
        t_closed: float = 60.0,
        t_open: float = 200.0,
    ) -> Tuple[float, float, int]:
        """
        5点ランドマーク(left_eye/right_eye)付近のパッチから
        ラプラシアン分散で「目が開いている度合い」を雑に推定。
        """
        try:
            lx, ly = landmarks.get("left_eye", (None, None))
            rx, ry = landmarks.get("right_eye", (None, None))
            if lx is None or rx is None:
                return 0.0, 0.0, 0

            x1, y1, x2, y2 = box
            bw = max(1, int(x2 - x1))
            ps = max(12, int(bw * float(patch_scale)))

            def _crop(cx: int, cy: int) -> np.ndarray:
                xx1 = max(0, cx - ps)
                yy1 = max(0, cy - ps)
                xx2 = min(int(image.shape[1]), cx + ps)
                yy2 = min(int(image.shape[0]), cy + ps)
                return image[yy1:yy2, xx1:xx2]

            patch_l = _crop(int(lx), int(ly))
            patch_r = _crop(int(rx), int(ry))

            def _lap_var(p: np.ndarray) -> float:
                if p.size == 0:
                    return 0.0
                gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                return float(lap.var())

            v_l = _lap_var(patch_l)
            v_r = _lap_var(patch_r)

            v = (v_l + v_r) / 2.0

            if float(t_open) <= float(t_closed):
                t_open = float(t_closed) + 1.0

            open_prob = (v - float(t_closed)) / (float(t_open) - float(t_closed))
            open_prob = float(np.clip(open_prob, 0.0, 1.0))
            closed_prob = 1.0 - open_prob

            return float(v), float(closed_prob), int(ps)
        except Exception:
            # 推定失敗時は「減点しない」寄り
            return 0.0, 0.0, 0
