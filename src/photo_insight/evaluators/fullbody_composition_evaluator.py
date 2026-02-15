import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from photo_insight.evaluators.base_composition_evaluator import BaseCompositionEvaluator
from photo_insight.utils.app_logger import Logger


class FullBodyCompositionEvaluator(BaseCompositionEvaluator):
    """
    全身ポーズや体のバランスに基づく構図評価クラス。

    - BODY_POSITION / BODY_BALANCE / POSE_DYNAMICS の 3 要素を 0〜1 で評価
    - 重み付き平均により body_composition_raw (0〜1) を算出
    - body_composition_score は Noise / Exposure と同じ 5 段階
      (0 / 0.25 / 0.5 / 0.75 / 1.0) の離散スコア
    """

    # スコアキー
    BODY_POSITION: str = "body_position_score"
    BODY_BALANCE: str = "body_balance_score"
    POSE_DYNAMICS: str = "pose_dynamics_score"
    FINAL_SCORE: str = "composition_fullbody_score"  # 従来互換用

    # 新しい意味スコア（全身構図のまとめ）
    BODY_COMPOSITION_RAW: str = "body_composition_raw"
    BODY_COMPOSITION_SCORE: str = "body_composition_score"

    # スコア分類閾値（グループ分類用・raw 用）
    HIGH_QUALITY_THRESHOLD: float = 0.8
    MEDIUM_QUALITY_THRESHOLD: float = 0.5

    # サブグループマッピング
    SUBGROUP_MAP: Dict[str, int] = {
        "high_quality": 2,
        "medium_quality": 1,
        "low_quality": 0,
    }

    # 重み設定（必要なら config で上書き）
    WEIGHTS: Dict[str, float] = {
        BODY_POSITION: 0.4,
        BODY_BALANCE: 0.3,
        POSE_DYNAMICS: 0.3,
    }

    def __init__(
        self,
        logger: Optional[Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        self.logger = logger or Logger(logger_name="FullBodyCompositionEvaluator")
        self.config = config or {}

        comp_conf = self.config.get("composition", {})
        disc_conf = comp_conf.get("discretize_thresholds", {})

        # 離散化用しきい値（0〜1）
        self.threshold_excellent = float(disc_conf.get("excellent", 0.85))
        self.threshold_good = float(disc_conf.get("good", 0.70))
        self.threshold_fair = float(disc_conf.get("fair", 0.55))
        self.threshold_poor = float(disc_conf.get("poor", 0.40))

        # 重みを config から上書き可能にしておく（任意）
        weights_conf: Dict[str, float] = comp_conf.get("fullbody_weights", {})
        if weights_conf:
            self.WEIGHTS = {
                **self.WEIGHTS,
                **{k: float(v) for k, v in weights_conf.items()},
            }

    def evaluate(
        self, image: np.ndarray, body_keypoints: List[Optional[List[float]]]
    ) -> Dict[str, float]:
        """
        画像とキーポイントから構図スコアを評価し、結果を返す。

        Returns:
            dict:
                - body_position_score / body_balance_score / pose_dynamics_score: 各0〜1の連続値
                - composition_fullbody_score: 従来の最終スコア (0〜1 連続)
                - body_composition_raw: 0〜1 の連続値（full body の意味スコア）
                - body_composition_score: 0 / 0.25 / 0.5 / 0.75 / 1.0 の離散値
                - group_id / subgroup_id: 従来通りのグループ分類
        """
        results: Dict[str, float] = {
            self.FINAL_SCORE: 0.0,
            self.BODY_POSITION: 0.0,
            self.BODY_BALANCE: 0.0,
            self.POSE_DYNAMICS: 0.0,
            "group_id": "unclassified",
            "subgroup_id": 0,
            # 新スキーマ
            self.BODY_COMPOSITION_RAW: None,
            self.BODY_COMPOSITION_SCORE: None,
        }

        results[self.BODY_POSITION] = self.evaluate_body_position(image, body_keypoints)
        results[self.BODY_BALANCE] = self.evaluate_body_balance(image, body_keypoints)
        results[self.POSE_DYNAMICS] = self.evaluate_pose_dynamics(image, body_keypoints)

        # 連続値の最終スコア（従来の composition_fullbody_score）
        final_raw: float = self.calculate_final_score(results)
        results[self.FINAL_SCORE] = final_raw
        results[self.BODY_COMPOSITION_RAW] = final_raw

        # 離散スコア（Noise / Exposure とスケールを揃える）
        results[self.BODY_COMPOSITION_SCORE] = self._to_discrete_score(final_raw)

        # グループ分類は raw スコアを使用（従来ロジック）
        group: str = self.classify_group(final_raw)
        results["group_id"] = group
        results["subgroup_id"] = self.SUBGROUP_MAP.get(group, 0)

        self.logger.debug(
            f"Scores - Position: {results[self.BODY_POSITION]:.2f}, "
            f"Balance: {results[self.BODY_BALANCE]:.2f}, "
            f"Dynamics: {results[self.POSE_DYNAMICS]:.2f}, "
            f"Raw: {final_raw:.2f}, "
            f"Discrete: {results[self.BODY_COMPOSITION_SCORE]:.2f}, "
            f"Group: {group}"
        )

        return results

    # -----------------------
    # 離散化ロジック
    # -----------------------
    def _to_discrete_score(self, value: float) -> float:
        """
        0〜1 の連続スコアを 0 / 0.25 / 0.5 / 0.75 / 1.0 に離散化する。
        Noise / Exposure / Composition でスケールを統一するための共通設計。
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
    # 個別スコア & グループ分類
    # -----------------------
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
        スコアに基づきグループ分類を行う（連続値ベース）。
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
