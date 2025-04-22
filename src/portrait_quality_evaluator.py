import numpy as np
import os
import traceback
import cv2
import gc

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
from image_loader import ImageLoader
from log_util import AppLogger
from utils.image_utils import ImageUtils


class PortraitQualityEvaluator:
    """
    ポートレート画像の品質を多角的に評価するためのクラス。
    顔認識、構図、シャープネス、ノイズ、コントラスト等を統合的にスコア化する。
    """

    def __init__(
        self,
        image_input: str | np.ndarray,
        is_raw: bool = False,
        logger: Optional[AppLogger] = None,
        file_name: Optional[str] = None,
        max_noise_value: float = 100.0,
        local_region_size: int = 32
    ):
        """
        クラスの初期化
        """
        self.is_raw = is_raw
        self.logger = logger or AppLogger(logger_name='PortraitQualityEvaluator')
        self.image_loader = ImageLoader(logger=self.logger)
        self.file_name = file_name if isinstance(image_input, np.ndarray) else os.path.basename(image_input)
        self.image_path = image_input if isinstance(image_input, str) else None

        self.rgb_image = self._load_image(image_input)
        self.resized_image_2048 = ImageUtils.resize_image(self.rgb_image, max_dimension=2048)
        self.resized_image_1024 = ImageUtils.resize_image(self.rgb_image, max_dimension=1024)

        # 評価器群（OCPに基づき、柔軟な拡張設計）
        self.evaluators = {
            "face": FaceEvaluator(backend='insightface'),
            "sharpness": SharpnessEvaluator(),
            "blurriness": BlurrinessEvaluator(),
            "contrast": ContrastEvaluator(),
            "noise": NoiseEvaluator(max_noise_value=max_noise_value),
            "local_sharpness": LocalSharpnessEvaluator(block_size=local_region_size),
            "local_contrast": LocalContrastEvaluator(block_size=local_region_size),
            "exposure": ExposureEvaluator(),  # 全体画像用
        }
        self.exposure_evaluator = ExposureEvaluator()  # 顔用（インスタンス分けてもOK）

        self.body_detector = FullBodyDetector()
        self.composition_evaluator = RuleBasedCompositionEvaluator(logger=self.logger)

        self.logger.info(f"画像ファイル {self.file_name} をロードしました")

    def _load_image(self, image_input: str | np.ndarray) -> np.ndarray:
        """画像の読み込み"""
        if isinstance(image_input, str):
            return self.image_loader.load_image(image_input, output_bps=16 if self.is_raw else 8)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("無効な入力タイプの画像データ")

    def evaluate(self) -> Dict[str, Any]:
        """
        総合評価実行
        """
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

            return results

        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}

    def _evaluate_face(self, evaluator, image):
        """
        顔領域の検出とスコアリング評価（視線情報含む）
        """
        def _empty_result():
            return {
                "face_sharpness_score": "",
                "face_contrast_score": "",
                "face_noise_score": "",
                "face_local_sharpness_score": "",
                "face_local_sharpness_std": "",
                "face_local_contrast_score": "",
                "face_local_contrast_std": "",
                "faces": [],
                "num_faces": 0,
                "face_detected": False,
                "yaw": "",
                "pitch": "",
                "roll": "",
                "gaze": ""
            }

        original_h, original_w = image.shape[:2]
        resized_image = self.resized_image_1024
        face_result = evaluator.evaluate(resized_image)

        if not face_result.get("face_detected", False):
            self.logger.warning("顔が検出されませんでした。")
            return _empty_result()

        faces = face_result.get("faces", [])
        refined_faces = []
        scale_factor = original_w / 1024

        for face in faces:
            x, y, w, h = [int(v * scale_factor) for v in face["box"]]
            padding = int(min(w, h) * 0.2)
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(original_w, x + w + padding), min(original_h, y + h + padding)
            cropped = image[y1:y2, x1:x2]

            if not isinstance(cropped, np.ndarray):
                self.logger.error("cropped_faceがnp.ndarrayではありません。スキップします。")
                continue

            sharpness = self._safe_evaluate(self.evaluators["sharpness"], cropped, "sharpness")
            contrast = self._safe_evaluate(self.evaluators["contrast"], cropped, "contrast")
            noise = self._safe_evaluate(self.evaluators["noise"], cropped, "noise")
            exposure = self._safe_evaluate(self.exposure_evaluator, cropped, "exposure")
            local_sharpness = self._safe_evaluate(self.evaluators["local_sharpness"], cropped, "local_sharpness")
            local_contrast = self._safe_evaluate(self.evaluators["local_contrast"], cropped, "local_contrast")

            refined_faces.append({
                "box": (x, y, w, h),
                "confidence": face.get("confidence", 0),
                "landmarks": face.get("landmarks", {}),
                "yaw": face.get("yaw", 0.0),
                "pitch": face.get("pitch", 0.0),
                "roll": face.get("roll", 0.0),
                "gaze": face.get("gaze"),
                **sharpness,
                **contrast,
                **noise,
                **local_sharpness,
                **local_contrast,
                **exposure,  # ← 追加
            })

        face_result["faces"] = refined_faces
        face_result["num_faces"] = len(refined_faces)

        best_face = max(refined_faces, key=lambda f: f.get("confidence", 0), default=None)

        if best_face:
            self.logger.info(f"最も信頼性の高い顔（confidence={best_face['confidence']}）を採用")
            face_result["face_detected"] = True

            # face_ 接頭辞付きでまとめて記録
            face_result["face_sharpness_score"] = best_face.get("sharpness_score", 0)
            face_result["face_contrast_score"] = best_face.get("contrast_score", 0)
            face_result["face_noise_score"] = best_face.get("noise_score", 0)
            face_result["face_local_sharpness_score"] = best_face.get("local_sharpness_score", 0)
            face_result["face_local_sharpness_std"] = best_face.get("local_sharpness_std", 0)
            face_result["face_local_contrast_score"] = best_face.get("local_contrast_score", 0)
            face_result["face_local_contrast_std"] = best_face.get("local_contrast_std", 0)
            face_result["face_exposure_score"] = best_face.get("exposure_score", 0)
            face_result["face_mean_brightness"] = best_face.get("mean_brightness", 0)

            face_result["yaw"] = best_face.get("yaw")
            face_result["pitch"] = best_face.get("pitch")
            face_result["roll"] = best_face.get("roll")
            face_result["gaze"] = best_face.get("gaze")
        else:
            self.logger.warning("信頼度の高い顔が選択できませんでした。")
            return _empty_result()

        return face_result

    def _evaluate_composition(self, image: np.ndarray, face_boxes: list) -> Dict[str, Any]:
        """
        構図評価の実行（顔位置をもとにしたルールベース）
        """
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
        """
        各評価モジュールを安全に実行し、標準化されたキーで結果を返す。
        """
        try:
            result = evaluator.evaluate(image)
            output = {
                f"{name}_score": result.get(f"{name}_score", 0)
            }

            if "local_" in name:
                output[f"{name}_std"] = result.get(f"{name}_std", 0)

            # メモリを解放
            gc.collect()

            return output

        except Exception as e:
            self.logger.error(f"{name} 評価中にエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {f"{name}_score": 0}

