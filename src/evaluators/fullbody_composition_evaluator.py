import numpy as np
from typing import List, Optional, Tuple, Dict

from evaluators.rule_based_composition_evaluator import RuleBasedCompositionEvaluator
from evaluators.base_composition_evaluator import BaseCompositionEvaluator
from utils.app_logger import Logger


class FullBodyCompositionEvaluator(BaseCompositionEvaluator):
    """
    全身ポーズや体のバランスに基づく構図評価クラス。
    """

    # スコアキー
    BODY_POSITION: str = "body_position_score"
    BODY_BALANCE: str = "body_balance_score"
    POSE_DYNAMICS: str = "pose_dynamics_score"
    FINAL_SCORE: str = "composition_fullbody_score"

    # スコア分類閾値
    HIGH_QUALITY_THRESHOLD: float = 0.8
    MEDIUM_QUALITY_THRESHOLD: float = 0.5

    # サブグループマッピング
    SUBGROUP_MAP: Dict[str, int] = {
        "high_quality": 2,
        "medium_quality": 1,
        "low_quality": 0,
    }

    # 重み設定
    WEIGHTS: Dict[str, float] = {
        BODY_POSITION: 0.4,
        BODY_BALANCE: 0.3,
        POSE_DYNAMICS: 0.3,
    }

    def __init__(self, logger: Optional[Logger] = None) -> None:
        self.logger = logger or Logger(logger_name="FullBodyCompositionEvaluator")

    def evaluate(
        self, image: np.ndarray, body_keypoints: List[Optional[List[float]]]
    ) -> Dict[str, float]:
        """
        画像とキーポイントから構図スコアを評価し、結果を返す。
        """
        results: Dict[str, float] = {
            self.FINAL_SCORE: 0.0,
            self.BODY_POSITION: 0.0,
            self.BODY_BALANCE: 0.0,
            self.POSE_DYNAMICS: 0.0,
            "group_id": "unclassified",
            "subgroup_id": 0,
        }

        results[self.BODY_POSITION] = self.evaluate_body_position(image, body_keypoints)
        results[self.BODY_BALANCE] = self.evaluate_body_balance(image, body_keypoints)
        results[self.POSE_DYNAMICS] = self.evaluate_pose_dynamics(image, body_keypoints)

        results[self.FINAL_SCORE] = self.calculate_final_score(results)

        group: str = self.classify_group(results[self.FINAL_SCORE])
        results["group_id"] = group
        results["subgroup_id"] = self.SUBGROUP_MAP.get(group, 0)

        self.logger.debug(
            f"Scores - Position: {results[self.BODY_POSITION]:.2f}, "
            f"Balance: {results[self.BODY_BALANCE]:.2f}, "
            f"Dynamics: {results[self.POSE_DYNAMICS]:.2f}, "
            f"Final: {results[self.FINAL_SCORE]:.2f}, Group: {group}"
        )

        return results

    def calculate_final_score(self, results: Dict[str, float]) -> float:
        """
        個別スコアと重みに基づき、最終スコアを計算。
        """
        total_weight: float = sum(self.WEIGHTS.values())
        if total_weight == 0:
            self.logger.error("Total weight is zero. Cannot compute final score.")
            return 0.0

        weighted_sum: float = sum(
            results[key] * weight for key, weight in self.WEIGHTS.items()
        )
        return round(weighted_sum / total_weight, 4)

    def classify_group(self, score: float) -> str:
        """
        スコアに基づきグループ分類を行う。
        """
        if score >= self.HIGH_QUALITY_THRESHOLD:
            return "high_quality"
        elif score >= self.MEDIUM_QUALITY_THRESHOLD:
            return "medium_quality"
        else:
            return "low_quality"

    def evaluate_body_position(
        self, image: np.ndarray, body_keypoints: List[Optional[List[float]]]
    ) -> float:
        """
        全身のフレーム内適合度を評価。
        """
        if not self._validate_keypoints(body_keypoints):
            self.logger.warning("Invalid body keypoints for position evaluation.")
            return 0.0

        height, width = image.shape[:2]
        in_frame_count: int = 0
        valid_points: int = 0

        for kp in body_keypoints:
            if self._is_valid_point(kp):
                valid_points += 1
                if 0 <= kp[0] < width and 0 <= kp[1] < height:
                    in_frame_count += 1

        return (in_frame_count / valid_points) if valid_points else 0.0

    def evaluate_body_balance(
        self, image: np.ndarray, body_keypoints: List[Optional[List[float]]]
    ) -> float:
        """
        重心位置と左右対称性から体のバランスを評価。
        """
        if not self._validate_keypoints(body_keypoints, min_points=17):
            self.logger.warning("Invalid body keypoints for balance evaluation.")
            return 0.0

        width: int = image.shape[1]
        center_score: float = self._calculate_center_score(body_keypoints, width)
        symmetry_score: float = self._calculate_symmetry_score(body_keypoints, width)

        return round((center_score + symmetry_score) / 2, 4)

    def evaluate_pose_dynamics(
        self, image: np.ndarray, body_keypoints: List[Optional[List[float]]]
    ) -> float:
        """
        ポーズの動的表現度を評価。
        """
        if not self._validate_keypoints(body_keypoints, min_points=17):
            self.logger.warning("Invalid body keypoints for dynamics evaluation.")
            return 0.0

        limb_pairs: List[Tuple[int, int]] = [(5, 9), (6, 10), (11, 15), (12, 16)]
        total_len: float = 0.0
        valid_pairs: int = 0

        for a, b in limb_pairs:
            if self._is_valid_point(body_keypoints[a]) and self._is_valid_point(
                body_keypoints[b]
            ):
                total_len += self._distance(body_keypoints[a], body_keypoints[b])
                valid_pairs += 1

        if valid_pairs == 0:
            return 0.0

        avg_len: float = total_len / valid_pairs
        img_diag: float = np.linalg.norm(image.shape[:2])
        return min(avg_len / img_diag, 1.0)

    # --- ヘルパーメソッド ---

    def _validate_keypoints(
        self, keypoints: List[Optional[List[float]]], min_points: int = 1
    ) -> bool:
        if keypoints is None:
            return False
        valid_points = [kp for kp in keypoints if self._is_valid_point(kp)]
        return len(valid_points) >= min_points

    def _is_valid_point(self, point: Optional[List[float]]) -> bool:
        return point is not None and len(point) >= 2 and all(p >= 0 for p in point[:2])

    def _distance(self, p1: List[float], p2: List[float]) -> float:
        return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]))

    def _calculate_center_score(
        self, keypoints: List[Optional[List[float]]], width: int
    ) -> float:
        center_pts: List[Optional[List[float]]] = [
            keypoints[5],
            keypoints[6],
            keypoints[11],
            keypoints[12],
        ]
        valid_xs: List[float] = [pt[0] for pt in center_pts if self._is_valid_point(pt)]
        if not valid_xs:
            return 0.0
        avg_x: float = np.mean(valid_xs)
        deviation: float = abs(avg_x - width / 2) / (width / 2)
        return max(0.0, 1.0 - deviation)

    def _calculate_symmetry_score(
        self, keypoints: List[Optional[List[float]]], width: int
    ) -> float:
        symmetry_pairs: List[Tuple[int, int]] = [
            (5, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
        ]
        diffs: List[float] = []
        for left_idx, right_idx in symmetry_pairs:
            left, right = keypoints[left_idx], keypoints[right_idx]
            if self._is_valid_point(left) and self._is_valid_point(right):
                mirrored_right_x: float = width - right[0]
                diffs.append(abs(left[0] - mirrored_right_x))
        if not diffs:
            return 0.0
        avg_diff: float = np.mean(diffs)
        return max(0.0, 1.0 - (avg_diff / width))
