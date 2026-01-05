import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Optional, List


class FullBodyDetector:
    """
    MediaPipe Pose を使った全身（姿勢）検出。

    互換:
      - detect_full_body(image) -> bool は残す（既存コードが壊れない）

    追加:
      - detect(image) -> Dict: 全身判定 + pose指標 + 擬似bbox + 余白比率 + 切れリスク
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        visibility_thresh: float = 0.5,
        model_complexity: int = 1,
        static_image_mode: bool = True,
    ):
        """
        :param visibility_thresh: “全身”判定に使う visibility 閾値
        :param model_complexity: 0/1/2 (精度/速度)
        :param static_image_mode: True=静止画向け（推奨）
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.visibility_thresh = visibility_thresh

        # “全身”判定に使う主要点
        self.required_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]

    # -------------------------
    # 互換 API（既存のまま）
    # -------------------------
    def detect_full_body(self, image: np.ndarray) -> bool:
        """
        画像内で全身を検出する（互換用: bool返却）
        :param image: 画像 (BGR形式のNumPy配列)
        :return: 全身が検出された場合は True
        """
        out = self.detect(image)
        return bool(out.get("full_body_detected", False))

    # -------------------------
    # 新 API（評価で使う）
    # -------------------------
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        姿勢推定を実行し、全身判定+派生指標を返す
        :param image: 画像 (BGR形式のNumPy配列)
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be numpy.ndarray")

        H, W = image.shape[:2]
        if H < 60 or W < 60:
            # 小さすぎるとPoseが不安定
            return self._empty_result(cut_risk=1.0)

        # BGR → RGB に変換 (MediaPipeはRGB入力)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 姿勢推定の実行
        results = self.pose.process(image_rgb)

        # ランドマークが検出されたか確認
        if not results.pose_landmarks:
            return self._empty_result(cut_risk=1.0)

        landmarks = results.pose_landmarks.landmark

        # required の visibility を集計
        vis_list = [float(landmarks[lm].visibility) for lm in self.required_landmarks]
        pose_score = float(np.clip(np.mean(vis_list) * 100.0, 0.0, 100.0))
        full_body_detected = all(v >= float(self.visibility_thresh) for v in vis_list)

        # ランドマーク正規化座標（0..1） → ピクセル bbox を推定
        xs = [float(lm.x) for lm in landmarks]
        ys = [float(lm.y) for lm in landmarks]

        # 安全: 範囲外に出ることがあるので clip
        x1 = int(np.clip(min(xs) * W, 0, W - 1))
        x2 = int(np.clip(max(xs) * W, 0, W - 1))
        y1 = int(np.clip(min(ys) * H, 0, H - 1))
        y2 = int(np.clip(max(ys) * H, 0, H - 1))

        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        pose_bbox = [int(x1), int(y1), int(bw), int(bh)]

        # 余白（全身構図の成立に効く）
        top_margin = y1
        bottom_margin = H - (y1 + bh)
        left_margin = x1
        right_margin = W - (x1 + bw)

        headroom_ratio = float(top_margin / H)
        footroom_ratio = float(bottom_margin / H)
        side_margin_min_ratio = float(min(left_margin, right_margin) / W)

        # 切れリスク（簡易）: 余白が小さいほどリスク↑
        # 0.0=安全, 1.0=危険
        risk = 0.0
        thr = 0.02
        if headroom_ratio < thr:
            risk += 0.4
        if footroom_ratio < thr:
            risk += 0.4
        if side_margin_min_ratio < thr:
            risk += 0.2
        full_body_cut_risk = float(np.clip(risk, 0.0, 1.0))

        return {
            "full_body_detected": bool(full_body_detected),
            "pose_score": pose_score,
            # CSVで重いなら将来 body_x/body_y/body_w/body_h に分解してもOK
            "pose_bbox": pose_bbox,
            "headroom_ratio": headroom_ratio,
            "footroom_ratio": footroom_ratio,
            "side_margin_min_ratio": side_margin_min_ratio,
            "full_body_cut_risk": full_body_cut_risk,
        }

    def draw_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        全身検出結果を画像に描画する（デバッグ用途）
        :param image: 画像 (BGR形式のNumPy配列)
        :return: ランドマークを描画した画像
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            annotated_image = image.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            return annotated_image
        return image

    def _empty_result(self, cut_risk: float = 1.0) -> Dict[str, Any]:
        return {
            "full_body_detected": False,
            "pose_score": 0.0,
            "pose_bbox": None,
            "headroom_ratio": None,
            "footroom_ratio": None,
            "side_margin_min_ratio": None,
            "full_body_cut_risk": float(np.clip(cut_risk, 0.0, 1.0)),
        }
