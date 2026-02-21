from insightface.app import FaceAnalysis
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from .base_face_evaluator import BaseFaceDetector


class InsightFaceDetector(BaseFaceDetector):
    """
    InsightFace を使用した顔検出クラス。

    - バウンディングボックス、ランドマーク、姿勢情報（yaw, pitch, roll）を取得可能。
    - OCP（オープン・クローズド原則）を意識し、処理ロジックをメソッド単位で分離。
    """

    def __init__(self, confidence_threshold: float = 0.5, gpu: bool = True):
        """
        コンストラクタ

        :param confidence_threshold: 顔検出の信頼度のしきい値（この値以上の顔のみを採用）
        :param gpu: GPU を使用するかどうか（True: GPU使用, False: CPU使用）
        """
        super().__init__(confidence_threshold)
        self.app = FaceAnalysis(name="buffalo_l")  # モデル名は必要に応じて変更可能
        self.app.prepare(ctx_id=0 if gpu else -1)

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        顔検出を行い、検出された顔ごとに情報を抽出する。

        :param image: BGR形式のNumPy配列画像
        :return: 顔検出結果を含む辞書（顔一覧、検出有無、検出数）
        """
        try:
            faces_raw = self.app.get(image)
        except Exception as e:
            print(f"顔検出時のエラー: {e}")
            return {"faces": [], "face_detected": False, "num_faces": 0}

        results = []
        for face in faces_raw:
            if face.det_score >= self.confidence_threshold:
                box = self._extract_box(face)
                landmarks = self._extract_landmarks(face)
                yaw, pitch, roll = self._extract_pose(face)
                gaze_vector = self._estimate_gaze_vector(yaw, pitch)

                # ★ 追加：閉眼推定
                eye_lap_var, eye_closed_prob, eye_patch_size = (
                    self._estimate_eye_closed(
                        image=image,
                        box=box,
                        landmarks=landmarks,
                    )
                )

                results.append(
                    {
                        "box": box,
                        "confidence": float(face.det_score),
                        "landmarks": landmarks,
                        "yaw": yaw,
                        "pitch": pitch,
                        "roll": roll,
                        "gaze": gaze_vector,
                        # ---- add ----
                        "eye_lap_var": eye_lap_var,
                        "eye_closed_prob": eye_closed_prob,
                        "eye_patch_size": eye_patch_size,
                    }
                )

        return {
            "faces": results,
            "face_detected": bool(results),
            "num_faces": len(results),
        }

    def _extract_box(self, face) -> List[int]:
        """
        バウンディングボックス（顔の外接矩形）を抽出する。

        :param face: InsightFace で検出された顔オブジェクト
        :return: [x1, y1, x2, y2] の形式のリスト
        """
        return face.bbox.astype(int).tolist()

    def _extract_landmarks(self, face) -> Dict[str, Tuple[int, int]]:
        """
        顔の5点ランドマークを抽出する。

        :param face: InsightFace で検出された顔オブジェクト
        :return: 各ランドマーク（目・鼻・口）座標の辞書
        """
        landmarks = face.kps.astype(int).tolist()
        return {
            "left_eye": landmarks[0],
            "right_eye": landmarks[1],
            "nose": landmarks[2],
            "mouth_left": landmarks[3],
            "mouth_right": landmarks[4],
        }

    def _extract_pose(self, face) -> Tuple[float, float, float]:
        """
        顔の姿勢（yaw, pitch, roll）を抽出する。

        pose属性またはnormed_pose属性のいずれかを優先的に使用。

        :param face: InsightFace で検出された顔オブジェクト
        :return: (yaw, pitch, roll) のタプル
        """
        if hasattr(face, "pose") and face.pose is not None:
            return tuple(face.pose)
        elif hasattr(face, "normed_pose") and face.normed_pose is not None:
            return tuple(face.normed_pose)
        else:
            return 0.0, 0.0, 0.0  # 姿勢情報が無い場合はデフォルト値

    def _estimate_gaze_vector(self, yaw: float, pitch: float) -> Dict[str, float]:
        """
        顔姿勢（yaw, pitch）から視線ベクトル（単位ベクトル）を推定する。

        :param yaw: 顔の左右向き（度）
        :param pitch: 顔の上下向き（度）
        :return: {"x": ..., "y": ..., "z": ...}
        """
        # 度 → ラジアン
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)

        x = np.sin(yaw_rad)
        y = np.sin(pitch_rad)
        z = np.cos(yaw_rad) * np.cos(pitch_rad)

        return {"x": float(x), "y": float(y), "z": float(z)}

    def _estimate_eye_closed(
        self,
        *,
        image: np.ndarray,
        box: List[int],
        landmarks: Dict[str, Tuple[int, int]],
        # 調整パラメータ（まずは固定でOK）
        patch_scale: float = 0.18,  # bbox幅に対するpatchサイズ
        t_closed: float = 60.0,  # lap_var がこれ未満なら閉眼寄り
        t_open: float = 200.0,  # lap_var がこれ以上なら開眼寄り
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
            # patchは bbox幅に比例（顔が小さくても相対スケールが保てる）
            ps = max(12, int(bw * patch_scale))

            def _crop(cx: int, cy: int) -> np.ndarray:
                xx1 = max(0, cx - ps)
                yy1 = max(0, cy - ps)
                xx2 = min(image.shape[1], cx + ps)
                yy2 = min(image.shape[0], cy + ps)
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

            # 代表値：左右平均（片目だけ潰れてるケースも拾いたければ min にしてもOK）
            v = (v_l + v_r) / 2.0

            # v を [0..1] の open_prob に正規化して、closed_prob に反転
            if t_open <= t_closed:
                t_open = t_closed + 1.0

            open_prob = (v - t_closed) / (t_open - t_closed)
            open_prob = float(np.clip(open_prob, 0.0, 1.0))
            closed_prob = 1.0 - open_prob

            return v, closed_prob, ps
        except Exception:
            # 推定失敗時は「減点しない」寄り（安全側）
            return 0.0, 0.0, 0
