import numpy as np
from photo_insight.evaluators.base_composition_evaluator import BaseCompositionEvaluator
from photo_insight.utils.app_logger import Logger
from typing import Any, Dict, Optional


class RuleBasedCompositionEvaluator(BaseCompositionEvaluator):
    """
    ルールベースの構図評価クラス。

    顔の位置、フレーミング、顔の向き、顔のスケール、視線の向きの
    5つのルールに基づいて構図を評価し、スコアと分類情報を返します。

    - 各ルールのスコアは 0〜1 の連続値
    - 最終スコアは平均値 (face_composition_raw)
    - 離散スコア (face_composition_score) は Noise / Exposure と同じ
      0 / 0.25 / 0.5 / 0.75 / 1.0 スケール
    """

    # スコア定義キー（保守性・補完性向上）
    FACE_POSITION = "face_position_score"
    FRAMING = "framing_score"
    FACE_DIRECTION = "face_direction_score"
    FACE_SCALE = "face_scale_score"
    EYE_CONTACT = "eye_contact_score"
    FINAL_SCORE = "composition_rule_based_score"  # 従来互換用

    # 新しい意味スコア（顔構図のまとめ）
    FACE_COMPOSITION_RAW = "face_composition_raw"
    FACE_COMPOSITION_SCORE = "face_composition_score"

    MAX_YAW_ANGLE = 45
    FRAME_MARGIN_RATIO = 0.05

    def __init__(
        self,
        logger=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            logger: ロガー。指定がない場合は AppLogger を使用。
            config: 閾値・重み・離散化ルールなどの設定。
                    例:
                    composition:
                      discretize_thresholds:
                        excellent: 0.85
                        good: 0.70
                        fair: 0.55
                        poor: 0.40
        """
        self.logger = logger or Logger(logger_name="RuleBasedCompositionEvaluator")
        self.config = config or {}

        comp_conf = self.config.get("composition", {})
        disc_conf = comp_conf.get("discretize_thresholds", {})

        # 離散化用しきい値（0〜1）
        self.threshold_excellent = float(disc_conf.get("excellent", 0.85))
        self.threshold_good = float(disc_conf.get("good", 0.70))
        self.threshold_fair = float(disc_conf.get("fair", 0.55))
        self.threshold_poor = float(disc_conf.get("poor", 0.40))

        # 関数参照ベースのルール評価関数群（IDE補完・安全性向上）
        self.RULE_EVALUATORS = [
            self.evaluate_face_position,
            self.evaluate_framing,
            self.evaluate_face_direction,
            self.evaluate_face_scale,
            self.evaluate_eye_contact,
        ]

    def evaluate(self, image: np.ndarray, face_boxes: list) -> dict:
        """
        画像と顔検出結果から構図スコアを計算する。

        Args:
            image (np.ndarray): BGR画像
            face_boxes (list):
              [{"box": [x, y, w, h], "confidence": float, "yaw": float (optional)}]

        Returns:
            dict: スコアと分類結果
                - 各ルールのスコア (0〜1 の連続値)
                - composition_rule_based_score: 従来の平均スコア (連続値)
                - face_composition_raw: 連続値 (0〜1)
                - face_composition_score: 離散値 (0 / 0.25 / 0.5 / 0.75 / 1.0)
        """
        results = {
            self.FINAL_SCORE: 0.0,
            self.FACE_POSITION: 0.0,
            self.FRAMING: 0.0,
            self.FACE_DIRECTION: 0.0,
            self.FACE_SCALE: 0.0,
            self.EYE_CONTACT: 0.0,
            "group_id": "unclassified",
            "subgroup_id": 0,
            # 新スキーマ
            self.FACE_COMPOSITION_RAW: None,
            self.FACE_COMPOSITION_SCORE: None,
        }

        if not face_boxes:
            self.logger.warning("顔が検出されませんでした。構図評価をスキップします。")
            # 顔がない場合は「評価不能」として raw=None, score=None のまま返す
            return results

        # 各ルール評価
        for rule_fn in self.RULE_EVALUATORS:
            rule_fn(image, face_boxes, results)

        keys = [
            self.FACE_POSITION,
            self.FRAMING,
            self.FACE_DIRECTION,
            self.FACE_SCALE,
            self.EYE_CONTACT,
        ]
        scores = [results[k] for k in keys if results[k] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # 連続値（生値）としての最終スコア
        results[self.FINAL_SCORE] = float(avg_score)  # 互換用
        results[self.FACE_COMPOSITION_RAW] = float(avg_score)

        # 離散スコア（Noise / Exposure と同じスケール）
        results[self.FACE_COMPOSITION_SCORE] = self._to_discrete_score(avg_score)

        # グループ分類は従来通り連続値で判定
        results["group_id"] = self.classify_group(*[results[k] for k in keys])
        results["subgroup_id"] = (
            int(results[self.FACE_POSITION] * 100) * 10000
            + int(results[self.FACE_DIRECTION] * 100) * 100
            + int(results[self.FRAMING] * 100)
        )

        self.logger.debug(f"Composition rule-based raw score: {avg_score:.2f}")
        self.logger.debug(
            f"Face composition discrete score: "
            f"{results[self.FACE_COMPOSITION_SCORE]:.2f}"
        )
        self.logger.debug(
            f"Group ID: {results['group_id']}, Subgroup ID: {results['subgroup_id']}"
        )
        return results

    # -----------------------
    # 離散化ロジック
    # -----------------------
    def _to_discrete_score(self, value: float) -> float:
        """
        0〜1 の連続スコアを 0 / 0.25 / 0.5 / 0.75 / 1.0 に離散化する。
        Noise / Exposure とスケールを揃える前提。
        """
        v = float(value)

        if v >= self.threshold_excellent:
            return 1.0
        if v >= self.threshold_good:
            return 0.75
        if v >= self.threshold_fair:
            return 0.5
        if v >= self.threshold_poor:
            return 0.25
        return 0.0

    # -----------------------
    # 既存ロジック（そのまま）
    # -----------------------
    def classify_group(
        self, fp: float, fr: float, fd: float, fs: float, ec: float
    ) -> str:
        """
        各スコアに基づき構図グループを分類。
        """

        def label_position(score):
            return (
                "position=well"
                if score >= 0.8
                else "position=moderate" if score >= 0.5 else "position=off"
            )

        def label_direction(score):
            return (
                "direction=good"
                if score >= 0.8
                else "direction=ok" if score >= 0.5 else "direction=bad"
            )

        def label_scale(score):
            return (
                "scale=very_small"
                if score < 0.2
                else (
                    "scale=small"
                    if score < 0.5
                    else "scale=large" if score > 0.9 else "scale=ideal"
                )
            )

        def label_framing(score):
            return "framing=ok" if score >= 1 else "framing=problem"

        def label_eye_contact(score):
            return (
                "eye_contact=strong"
                if score >= 0.8
                else "eye_contact=weak" if score >= 0.5 else "eye_contact=none"
            )

        if all(score >= 0.8 for score in [fp, fr, fd, fs, ec]):
            return "composition=ideal"

        return " | ".join(
            [
                label_position(fp),
                label_direction(fd),
                label_scale(fs),
                label_framing(fr),
                label_eye_contact(ec),
            ]
        )

    def evaluate_face_position(
        self, image: np.ndarray, face_boxes: list, results: dict
    ):
        """
        顔の位置スコア（三分割法との距離）を計算。
        """
        height, width = image.shape[:2]
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))

        try:
            x, y, w, h = main_face["box"]
        except KeyError as e:
            self.logger.error(f"main_face に必要なキーがありません: {e}")
            return
        except (ValueError, TypeError) as e:
            self.logger.error(f"main_face の box フォーマットが不正です: {e}")
            return

        face_center_x = x + w / 2
        face_center_y = y + h / 2

        thirds = [
            (width / 3, height / 3),
            (2 * width / 3, height / 3),
            (width / 3, 2 * height / 3),
            (2 * width / 3, 2 * height / 3),
        ]
        distances = [
            np.hypot(face_center_x - tx, face_center_y - ty) for tx, ty in thirds
        ]
        max_distance = np.hypot(width / 3, height / 3)
        score = 1 - min(min(distances) / max_distance, 1)
        results[self.FACE_POSITION] = score

        self.logger.debug(f"Face center: ({face_center_x:.1f}, {face_center_y:.1f})")
        self.logger.debug(f"Face position score: {score:.2f}")

    def evaluate_framing(self, image: np.ndarray, face_boxes: list, results: dict):
        """
        顔がフレームに収まっているかを評価。
        """
        height, width = image.shape[:2]
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))

        try:
            x, y, w, h = main_face["box"]
        except KeyError as e:
            self.logger.error(f"main_face に必要なキーがありません: {e}")
            return
        except (ValueError, TypeError) as e:
            self.logger.error(f"main_face の box フォーマットが不正です: {e}")
            return

        margin = self.FRAME_MARGIN_RATIO
        x_in = -w * margin <= x <= width - (1 - margin) * w
        y_in = -h * margin <= y <= height - (1 - margin) * h
        results[self.FRAMING] = 1 if x_in and y_in else 0

        self.logger.debug(f"Framing score: {results[self.FRAMING]}")

    def evaluate_face_direction(
        self, image: np.ndarray, face_boxes: list, results: dict
    ):
        """
        顔の向き（yaw）から正面度をスコア化。
        """
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))
        yaw = main_face.get("yaw")

        if yaw is not None:
            score = max(0, 1 - abs(yaw) / self.MAX_YAW_ANGLE)
            results[self.FACE_DIRECTION] = score
            self.logger.debug(f"Face direction score: {score:.2f}")
        else:
            self.logger.warning("顔の向き情報（yaw）が見つかりません。")
            results[self.FACE_DIRECTION] = 0

    def evaluate_face_scale(self, image: np.ndarray, face_boxes: list, results: dict):
        """
        顔のサイズ（面積比）からスケールの適切さを評価。
        """
        height, width = image.shape[:2]
        image_area = width * height
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))

        try:
            x, y, w, h = main_face["box"]
            face_area = w * h
            ratio = face_area / image_area
            ideal = 0.15
            tolerance = 0.1
            score = max(0, 1 - abs(ratio - ideal) / tolerance)
            results[self.FACE_SCALE] = round(score, 2)
            self.logger.debug(f"Face scale ratio: {ratio:.3f}, score: {score:.2f}")
        except KeyError as e:
            self.logger.warning(f"顔サイズ評価中に必要なキーが不足: {e}")
            results[self.FACE_SCALE] = 0
        except Exception as e:
            self.logger.warning(f"顔サイズ評価中にエラー: {e}")
            results[self.FACE_SCALE] = 0

    def evaluate_eye_contact(self, image: np.ndarray, face_boxes: list, results: dict):
        """
        視線がカメラ正面を向いているかどうかをスコア化。
        """
        main_face = max(face_boxes, key=lambda f: f.get("confidence", 0))
        gaze = main_face.get("gaze")

        if gaze:
            gaze_x = gaze.get("x", 0)
            gaze_y = gaze.get("y", 0)
            gaze_z = gaze.get("z", 0)
            norm = np.linalg.norm([gaze_x, gaze_y, gaze_z])
            forward_ratio = gaze_z / norm if norm != 0 else 0
            score = max(0, min(1, forward_ratio))
            results[self.EYE_CONTACT] = round(score, 2)
            self.logger.debug(f"Eye contact score: {score:.2f}")
        else:
            self.logger.warning("視線情報（gaze）が見つかりません。")
            results[self.EYE_CONTACT] = 0
