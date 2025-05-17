import numpy as np
import os
import traceback
import gc
import cv2

from typing import Optional, Tuple, Dict, Any

# 自作ライブラリ（変更不可）
from evaluators.face_evaluator import FaceEvaluator
from evaluators.sharpness_evaluator import SharpnessEvaluator
from evaluators.blurriness_evaluator import BlurrinessEvaluator
from evaluators.contrast_evaluator import ContrastEvaluator
from evaluators.noise_evaluator import NoiseEvaluator
from evaluators.exposure_evaluator import ExposureEvaluator
from evaluators.local_sharpness_evaluator import LocalSharpnessEvaluator
from evaluators.local_contrast_evaluator import LocalContrastEvaluator
from evaluators.rule_based_composition_evaluator import RuleBasedCompositionEvaluator
from detectors.body_detection import FullBodyDetector
from utils.app_logger import Logger
from image_utils.image_preprocessor import ImagePreprocessor


class PortraitQualityEvaluator:
    """
    ポートレート画像の品質を多角的に評価するためのクラス。
    顔認識、構図、シャープネス、ノイズ、コントラスト等を統合的にスコア化する。
    """

    def __init__(
        self,
        image_input: str | np.ndarray,
        is_raw: bool = False,
        logger: Optional[Logger] = None,
        file_name: Optional[str] = None,
        max_noise_value: float = 100.0,
        local_region_size: int = 32
    ):
        self.is_raw = is_raw
        self.logger = logger or Logger(logger_name='PortraitQualityEvaluator')
        self.file_name = file_name if isinstance(image_input, np.ndarray) else os.path.basename(image_input)
        self.image_path = image_input if isinstance(image_input, str) else None

        # 前処理器で画像を一括取得
        self.preprocessor = ImagePreprocessor(logger=self.logger, is_raw=self.is_raw, gamma=1.2)
        images = self.preprocessor.load_and_resize(image_input)
        self.rgb_image = images["original"]
        self.resized_image_2048 = images["resized_2048"]
        self.resized_image_1024 = images["resized_1024"]

        self.evaluators = {
            "face": FaceEvaluator(backend='insightface'),
            "sharpness": SharpnessEvaluator(),
            "blurriness": BlurrinessEvaluator(),
            "contrast": ContrastEvaluator(),
            "noise": NoiseEvaluator(max_noise_value=max_noise_value),
            "local_sharpness": LocalSharpnessEvaluator(block_size=local_region_size),
            "local_contrast": LocalContrastEvaluator(block_size=local_region_size),
            "exposure": ExposureEvaluator(),
        }

        self.body_detector = FullBodyDetector()
        self.composition_evaluator = RuleBasedCompositionEvaluator(logger=self.logger)

        self.logger.info(f"画像ファイル {self.file_name} をロードしました")

    def evaluate(self) -> Dict[str, Any]:
        self.logger.info(f"評価開始: 画像ファイル {self.file_name}")
        results = {}

        try:
            if self.resized_image_2048 is None:
                self.logger.error("resized_image_2048 が None です。評価をスキップします。")
                return {}

            face_result = self._evaluate_face(self.evaluators["face"], self.rgb_image)
            results.update(face_result)

            composition_result = self._evaluate_composition(self.rgb_image, face_result.get("faces", []))
            results.update(composition_result)

            for key, evaluator in self.evaluators.items():
                if key == "face":
                    continue
                results.update(self._safe_evaluate(evaluator, self.resized_image_2048, key))

            # 最も信頼性の高い顔に対する局所評価と属性追加
            if face_result.get("faces"):
                best_face = max(face_result["faces"], key=lambda f: f.get("confidence", 0))
                box = best_face.get("box") or best_face.get("bbox")
                if box and len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    face_crop = self.rgb_image[y1:y2, x1:x2]
                    results.update(self._evaluate_face_region(face_crop))

                # 追加属性セット
                for attr in ["yaw", "pitch", "roll", "gaze"]:
                    if attr in best_face:
                        results[attr] = best_face[attr]

            return results

        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}

    def _evaluate_face(self, evaluator, image):
        try:
            result = evaluator.evaluate(image)
            self.logger.info(f"顔評価結果: {result}")
            return result
        except Exception as e:
            self.logger.warning(f"顔評価中にエラー: {str(e)}")
            return {"face_score": 0, "faces": []}

    def _evaluate_composition(self, image: np.ndarray, face_boxes: list) -> Dict[str, Any]:
        try:
            if not face_boxes:
                self.logger.warning("構図評価スキップ：face_boxes が空です。")
                return {
                    "composition_rule_based_score": 0,
                    "face_position_score": 0,
                    "framing_score": 0,
                    "face_direction_score": 0,
                    "group_id": -1,
                    "subgroup_id": -1
                }

            result = self.composition_evaluator.evaluate(image, face_boxes)
            self.logger.info(f"構図評価結果: {result}")
            return {
                "composition_rule_based_score": result.get("composition_rule_based_score", 0),
                "face_position_score": result.get("face_position_score", 0),
                "framing_score": result.get("framing_score", 0),
                "face_direction_score": result.get("face_direction_score", 0),
                "group_id": result.get("group_id", -1),
                "subgroup_id": result.get("subgroup_id", -1),
            }

        except Exception as e:
            self.logger.warning(f"構図評価中にエラー: {str(e)}")
            return {
                "composition_rule_based_score": 0,
                "face_position_score": 0,
                "framing_score": 0,
                "face_direction_score": 0,
                "group_id": -1,
                "subgroup_id": -1,
            }

    def _safe_evaluate(self, evaluator, image: np.ndarray, name: str) -> Dict[str, Any]:
        try:
            result = evaluator.evaluate(image)
            output = {
                f"{name}_score": result.get(f"{name}_score", 0)
            }

            if "local_" in name:
                output[f"{name}_std"] = result.get(f"{name}_std", 0)

            if name == "exposure":
                output["mean_brightness"] = result.get("mean_brightness", 0)

            gc.collect()
            return output

        except Exception as e:
            self.logger.error(f"{name} 評価中にエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {f"{name}_score": 0}

    def _evaluate_face_region(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        顔領域に対して個別評価器を適用し、局所スコアを算出する。
        """
        face_scores = {}
        for key in ["sharpness", "contrast", "noise", "local_sharpness", "local_contrast", "exposure"]:
            evaluator = self.evaluators.get(key)
            if evaluator is None:
                continue
            try:
                result = evaluator.evaluate(face_crop)
                score_key = f"face_{key}_score"
                face_scores[score_key] = result.get(f"{key}_score", 0)

                if "local_" in key:
                    face_scores[f"face_{key}_std"] = result.get(f"{key}_std", 0)

                if key == "exposure":
                    face_scores["face_mean_brightness"] = result.get("mean_brightness", 0)

            except Exception as e:
                self.logger.warning(f"顔領域の{key}評価に失敗: {str(e)}")
                face_scores[f"face_{key}_score"] = 0

        return face_scores
