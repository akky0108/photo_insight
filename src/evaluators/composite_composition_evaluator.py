from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from evaluators.rule_based_composition_evaluator import RuleBasedCompositionEvaluator
from evaluators.fullbody_composition_evaluator import FullBodyCompositionEvaluator
from evaluators.base_composition_evaluator import BaseCompositionEvaluator
from utils.app_logger import Logger


class CompositeCompositionEvaluator(BaseCompositionEvaluator):
    """
    顔構図評価と全身構図評価を統合するファサードクラス。

    - RuleBasedCompositionEvaluator: 顔中心の構図評価
        - face_composition_raw / face_composition_score を返す
    - FullBodyCompositionEvaluator: 全身の構図評価
        - body_composition_raw / body_composition_score を返す

    本クラスではそれらを統合して:
        - composition_raw  : 顔＋全身の連続スコア (0〜1)
        - composition_score: 離散スコア (0 / 0.25 / 0.5 / 0.75 / 1.0)
        - composition_status: face_only / body_only / face_and_body / not_computed...
    を返す。
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            logger: 任意のロガーインスタンス。指定がない場合は AppLogger を使用。
            config: 閾値・重み・離散化ルールなどの設定。
                    例:
                    composition:
                      face_weight: 1.0
                      body_weight: 1.0
                      discretize_thresholds:
                        excellent: 0.85
                        good: 0.70
                        fair: 0.55
                        poor: 0.40
        """
        self.logger = logger or Logger(logger_name="CompositeCompositionEvaluator")

        self.config = config or {}
        comp_conf = self.config.get("composition", {})

        # 顔・全身スコアの統合重み
        self.face_weight: float = float(comp_conf.get("face_weight", 1.0))
        self.body_weight: float = float(comp_conf.get("body_weight", 1.0))

        # 離散化用しきい値（0〜1）
        disc_conf = comp_conf.get("discretize_thresholds", {})
        self.threshold_excellent: float = float(disc_conf.get("excellent", 0.85))
        self.threshold_good: float = float(disc_conf.get("good", 0.70))
        self.threshold_fair: float = float(disc_conf.get("fair", 0.55))
        self.threshold_poor: float = float(disc_conf.get("poor", 0.40))

        self.face_evaluator = RuleBasedCompositionEvaluator(
            logger=self.logger,
            config=self.config,
        )
        self.body_evaluator = FullBodyCompositionEvaluator(
            logger=self.logger,
            config=self.config,
        )

    def evaluate(
        self,
        image: np.ndarray,
        face_boxes: List[Dict[str, Any]],
        body_keypoints: List[Optional[List[float]]],
    ) -> Dict[str, Any]:
        """
        顔と全身の構図評価を統合して実行。

        Args:
            image: 評価対象画像 (H, W, C)
            face_boxes: 顔検出結果リスト
            body_keypoints: 全身キーポイントリスト

        Returns:
            dict: 統合された評価結果。
                  - face_composition_raw / face_composition_score
                  - body_composition_raw / body_composition_score
                  - composition_raw / composition_score
                  - composition_status
                  加えて、各 Evaluator が返す他のキーもそのまま含まれる。
        """
        self.logger.info("Starting composite composition evaluation.")

        face_results: Dict[str, Any] = self.face_evaluator.evaluate(image, face_boxes) or {}
        body_results: Dict[str, Any] = self.body_evaluator.evaluate(image, body_keypoints) or {}

        # 個別スコアを取得（なければ None）
        face_raw = face_results.get("face_composition_raw")
        body_raw = body_results.get("body_composition_raw")

        # 統合生値を計算
        composition_raw, status = self._merge_raw_scores(face_raw, body_raw)

        # 離散スコア化
        if composition_raw is None:
            # 「分からないのでニュートラル」として 0.5 を付与
            composition_score = 0.5
            if status == "not_computed":
                status = "not_computed_with_default"
        else:
            composition_score = self._to_discrete_score(composition_raw)

        # 顔・全身の離散スコアも、もし raw のみならここで付与しておく
        face_score = face_results.get("face_composition_score")
        if face_score is None and face_raw is not None:
            face_score = self._to_discrete_score(face_raw)

        body_score = body_results.get("body_composition_score")
        if body_score is None and body_raw is not None:
            body_score = self._to_discrete_score(body_raw)

        combined_results: Dict[str, Any] = {
            **face_results,
            **body_results,  # 注意: group_id / subgroup_id は全身側が上書きする
            "face_composition_raw": face_raw,
            "face_composition_score": face_score,
            "body_composition_raw": body_raw,
            "body_composition_score": body_score,
            "composition_raw": composition_raw,
            "composition_score": composition_score,
            "composition_status": status,
        }

        self.logger.info(
            f"Composite composition evaluation completed. "
            f"raw={composition_raw}, score={composition_score}, status={status}"
        )
        return combined_results

    # -----------------------
    # 内部ヘルパー
    # -----------------------
    def _merge_raw_scores(
        self,
        face_raw: Optional[float],
        body_raw: Optional[float],
    ) -> Tuple[Optional[float], str]:
        """
        顔構図・全身構図の生値を統合して 0〜1 の連続スコアを作る。

        戻り値:
            (composition_raw, status)
            status: "not_computed" / "face_only" / "body_only" / "face_and_body" ...
        """
        # どちらも None → 計算不能
        if face_raw is None and body_raw is None:
            return None, "not_computed"

        # 片方だけあるパターン + 「body_raw が 0.0 なら信用しない」みたいな扱い
        if face_raw is not None and (body_raw is None or body_raw <= 0.0):
            return float(face_raw), "face_only"

        if body_raw is not None and (face_raw is None or face_raw <= 0.0):
            return float(body_raw), "body_only"

        # 両方ある場合 → 重み付き平均
        w_face = self.face_weight
        w_body = self.body_weight
        if w_face <= 0 and w_body <= 0:
            # さすがにこれはないと思うが安全側に倒す
            raw = (float(face_raw) + float(body_raw)) / 2.0
            return raw, "both_zero_weight"

        raw = (float(face_raw) * w_face + float(body_raw) * w_body) / (w_face + w_body)
        return raw, "face_and_body"

    def _to_discrete_score(self, value: float) -> float:
        """
        0〜1 の連続スコアを 0 / 0.25 / 0.5 / 0.75 / 1.0 に離散化する。

        閾値は config から取得。指定がなければデフォルト値を使用。
        """
        if value is None:
            # 呼び出し側で None チェックする設計だが、一応保険としてニュートラルを返す
            return 0.5

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
