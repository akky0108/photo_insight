from evaluators.rule_based_composition_evaluator import RuleBasedCompositionEvaluator
from evaluators.fullbody_composition_evaluator import FullBodyCompositionEvaluator
from evaluators.base_composition_evaluator import BaseCompositionEvaluator
from utils.app_logger import Logger
import numpy as np


class CompositeCompositionEvaluator(BaseCompositionEvaluator):
    """
    顔構図評価と全身構図評価を統合するファサードクラス。

    単一のインターフェースから両評価を実行し、結果を統合して返す。
    使用する評価クラス:
        - RuleBasedCompositionEvaluator: 顔中心の構図評価
        - FullBodyCompositionEvaluator: 全身の構図評価（立ち位置、姿勢、バランスなど）
    """

    def __init__(self, logger=None):
        """
        コンストラクタ。

        Args:
            logger: 任意のロガーインスタンス。指定がない場合はAppLoggerを使用。
        """
        self.logger = logger or Logger("CompositeCompositionEvaluator")
        self.face_evaluator = RuleBasedCompositionEvaluator(logger=self.logger)
        self.body_evaluator = FullBodyCompositionEvaluator(logger=self.logger)

    def evaluate(
        self, image: np.ndarray, face_boxes: list, body_keypoints: list
    ) -> dict:
        """
        顔と全身の構図評価を統合して実行。

        Args:
            image (np.ndarray): 入力画像
            face_boxes (list): 顔検出情報のリスト
            body_keypoints (list): 全身ポーズ推定のキーポイント情報

        Returns:
            dict: 統合された評価結果
        """
        self.logger.info("Starting composite composition evaluation.")

        face_results = self.face_evaluator.evaluate(image, face_boxes)
        body_results = self.body_evaluator.evaluate(image, body_keypoints)

        # 結果を統合（キーが重複する場合は body 側で上書き）
        combined_results = {**face_results, **body_results}

        self.logger.info("Composite composition evaluation completed.")
        return combined_results
