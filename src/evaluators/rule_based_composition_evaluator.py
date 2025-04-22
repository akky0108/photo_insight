import numpy as np
from evaluators.base_composition_evaluator import BaseCompositionEvaluator
from log_util import AppLogger

class RuleBasedCompositionEvaluator(BaseCompositionEvaluator):
    """
    ルールベースの構図評価クラス。

    顔の位置、フレーミング、顔の向きの3つの観点から構図を評価し、
    類似構図のグループ化と相対ランク付けのためのIDも返す。
    """

    MAX_YAW_ANGLE = 45          # 顔の向きが正面からどれだけ逸脱するとスコア0になるか
    FRAME_MARGIN_RATIO = 0.05   # 画像外に出ても許容する余白の比率

    RULE_EVALUATORS = [
        "evaluate_face_position",
        "evaluate_framing",
        "evaluate_face_direction",
        "evaluate_face_scale",
        "evaluate_eye_contact"
    ]

    def __init__(self, logger=None):
        self.logger = logger or AppLogger(logger_name='RuleBasedCompositionEvaluator')

    def evaluate(self, image: np.ndarray, face_boxes: list) -> dict:
        """
        顔検出結果と画像から構図評価スコアを算出。

        Args:
            image (np.ndarray): 評価対象画像（BGR）
            face_boxes (list): [{"box": [x, y, w, h], "confidence": float, "yaw": float (optional)}]

        Returns:
            dict: 評価結果（スコア、グループID、ランクID）
        """
        results = {
            "composition_rule_based_score": 0,
            "face_position_score": 0,
            "framing_score": 0,
            "face_direction_score": 0,
            "eye_contact_score": 0,
            "group_id": "unclassified",
            "subgroup_id": 0
        }

        if not face_boxes:
            self.logger.warning("顔が検出されませんでした。構図評価をスキップします。")
            return results

        for rule in self.RULE_EVALUATORS:
            rule_fn = getattr(self, rule)
            rule_fn(image, face_boxes, results)

        # 平均スコア算出
        keys = [
            "face_position_score",
            "framing_score",
            "face_direction_score",
            "face_scale_score",
            "eye_contact_score"
        ]
        scores = [results[k] for k in keys if results[k] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        results["composition_rule_based_score"] = avg_score

        # グループ分類
        fp = results["face_position_score"]
        fr = results["framing_score"]
        fd = results["face_direction_score"]
        fs = results["face_scale_score"]
        ec = results["eye_contact_score"]
        results["group_id"] = self.classify_group(fp, fr, fd, fs, ec)

        # サブグループID: スコア構成 (fp * 10000 + fd * 100 + fr)
        results["subgroup_id"] = (
            int(fp * 100) * 10000 +
            int(fd * 100) * 100 +
            int(fr * 100)
        )

        self.logger.debug(f"Composition rule-based score: {avg_score:.2f}")
        self.logger.debug(f"Group ID: {results['group_id']}, Subgroup ID: {results['subgroup_id']}")
        return results


    def classify_group(self, fp: float, fr: float, fd: float, fs: float, ec: float) -> str:
        def label_position(score):
            if score >= 0.8:
                return "position=well"
            elif score >= 0.5:
                return "position=moderate"
            else:
                return "position=off"

        def label_direction(score):
            if score >= 0.8:
                return "direction=good"
            elif score >= 0.5:
                return "direction=ok"
            else:
                return "direction=bad"

        def label_scale(score):
            if score < 0.2:
                return "scale=very_small"
            elif score < 0.5:
                return "scale=small"
            elif score > 0.9:
                return "scale=large"
            else:
                return "scale=ideal"

        def label_framing(score):
            return "framing=ok" if score >= 1 else "framing=problem"

        def label_eye_contact(score):
            if score >= 0.8:
                return "eye_contact=strong"
            elif score >= 0.5:
                return "eye_contact=weak"
            else:
                return "eye_contact=none"

        if all(score >= 0.8 for score in [fp, fr, fd, fs, ec]):
            return "composition=ideal"

        labels = [
            label_position(fp),
            label_direction(fd),
            label_scale(fs),
            label_framing(fr),
            label_eye_contact(ec)
        ]

        return " | ".join(labels)


    def evaluate_face_position(self, image: np.ndarray, face_boxes: list, results: dict):
        """
        顔の位置を三分割法に基づき評価（黄金位置との距離）。
        """
        height, width = image.shape[:2]
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))
        try:
            x, y, w, h = main_face["box"]
        except (KeyError, ValueError, TypeError):
            self.logger.error("main_face に不正な box 情報があります。")
            return

        face_center_x = x + w / 2
        face_center_y = y + h / 2

        third_width = width / 3
        third_height = height / 3
        target_points = [
            (third_width, third_height),
            (2 * third_width, third_height),
            (third_width, 2 * third_height),
            (2 * third_width, 2 * third_height)
        ]

        distances = [np.hypot(face_center_x - tx, face_center_y - ty) for tx, ty in target_points]
        max_distance = np.hypot(third_width, third_height)
        results["face_position_score"] = 1 - min(min(distances) / max_distance, 1)
        self.logger.debug(f"Face center: ({face_center_x:.1f}, {face_center_y:.1f})")
        self.logger.debug(f"Face position score: {results['face_position_score']:.2f}")

    def evaluate_framing(self, image: np.ndarray, face_boxes: list, results: dict):
        """
        顔がフレームに収まっているかを評価。
        """
        height, width = image.shape[:2]
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))
        try:
            x, y, w, h = main_face["box"]
        except (KeyError, ValueError, TypeError):
            self.logger.error("main_face に不正な box 情報があります。")
            return

        margin = self.FRAME_MARGIN_RATIO
        x_in_frame = -w * margin <= x <= width - (1 - margin) * w
        y_in_frame = -h * margin <= y <= height - (1 - margin) * h

        results["framing_score"] = 1 if x_in_frame and y_in_frame else 0
        self.logger.debug(f"Framing score: {results['framing_score']}")

    def evaluate_face_direction(self, image: np.ndarray, face_boxes: list, results: dict):
        """
        顔の向きを評価（正面に近いほど高スコア）。
        """
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))
        yaw = main_face.get("yaw", None)
        if yaw is not None:
            results["face_direction_score"] = max(0, 1 - abs(yaw) / self.MAX_YAW_ANGLE)
            self.logger.debug(f"Face direction score: {results['face_direction_score']:.2f}")
        else:
            self.logger.warning("顔の向き情報（yaw）が見つかりません。")
            results["face_direction_score"] = 0

    def evaluate_face_scale(self, image: np.ndarray, face_boxes: list, results: dict):
        """
        顔サイズの評価。画像全体に対する顔の面積比をもとに、理想的なスケールかどうかを判定。
        """
        height, width = image.shape[:2]
        image_area = width * height

        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))
        try:
            x, y, w, h = main_face["box"]
            face_area = w * h
            face_ratio = face_area / image_area  # 顔面積の画像比率

            ideal_ratio = 0.15  # 理想は画像の15%
            tolerance = 0.1     # ±10%までは許容範囲

            score = max(0, 1 - abs(face_ratio - ideal_ratio) / tolerance)
            results["face_scale_score"] = round(score, 2)
            self.logger.debug(f"Face scale ratio: {face_ratio:.3f}, score: {score:.2f}")
        except Exception as e:
            self.logger.warning(f"顔サイズ評価中にエラー: {e}")
            results["face_scale_score"] = 0

    def evaluate_eye_contact(self, image: np.ndarray, face_boxes: list, results: dict):
        """
        視線がカメラに向いているかを評価。
        """
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))
        gaze = main_face.get("gaze", None)  # gaze = {"x": ..., "y": ..., "z": ...}

        if gaze:
            gaze_x = gaze.get("x", 0)
            gaze_y = gaze.get("y", 0)
            gaze_z = gaze.get("z", 0)
            norm = np.linalg.norm([gaze_x, gaze_y, gaze_z])
            if norm == 0:
                score = 0
            else:
                forward_ratio = gaze_z / norm
                score = max(0, min(1, forward_ratio))
            results["eye_contact_score"] = round(score, 2)
            self.logger.debug(f"Eye contact score: {results['eye_contact_score']:.2f}")
        else:
            self.logger.warning("視線情報（gaze）が見つかりません。")
            results["eye_contact_score"] = 0

