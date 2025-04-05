import cv2
import numpy as np
import mediapipe as mp

class FullBodyDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        全身検出器を初期化
        :param min_detection_confidence: 検出の信頼度の閾値
        :param min_tracking_confidence: トラッキングの信頼度の閾値
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect_full_body(self, image: np.ndarray) -> bool:
        """
        画像内で全身を検出する
        :param image: 画像 (RGB形式のNumPy配列)
        :return: 全身が検出された場合は True, それ以外は False
        """
        # BGR → RGB に変換 (MediaPipeはRGB入力)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 姿勢推定の実行
        results = self.pose.process(image_rgb)
        
        # ランドマークが検出されたか確認
        if not results.pose_landmarks:
            return False  # 姿勢が検出されなければFalse
        
        # ランドマークを取得
        landmarks = results.pose_landmarks.landmark

        # 主要なランドマーク（頭部・肩・腰・足）の存在確認
        required_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,       # 鼻（頭部）
            self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,  # 両肩
            self.mp_pose.PoseLandmark.LEFT_HIP, 
            self.mp_pose.PoseLandmark.RIGHT_HIP,      # 両腰
            self.mp_pose.PoseLandmark.LEFT_ANKLE, 
            self.mp_pose.PoseLandmark.RIGHT_ANKLE     # 両足首
        ]

        # すべてのランドマークが検出されているか
        if all(landmarks[lm].visibility > 0.5 for lm in required_landmarks):
            return True  # 全身が検出されたと判断

        return False  # 一部のランドマークが見えていない場合は全身とは判定しない

    def draw_landmarks(self, image: np.ndarray) -> np.ndarray:
        """
        全身検出結果を画像に描画する
        :param image: 画像 (BGR形式のNumPy配列)
        :return: ランドマークを描画した画像
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            # ランドマークを描画
            annotated_image = image.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            return annotated_image
        return image  # ランドマークが検出されなかった場合はそのまま返す
