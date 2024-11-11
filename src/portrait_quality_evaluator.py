import json
import numpy as np
import os
import traceback
from evaluators.face_evaluator import FaceEvaluator
from evaluators.sharpness_evaluator import SharpnessEvaluator
from evaluators.blurriness_evaluator import BlurrinessEvaluator
from evaluators.contrast_evaluator import ContrastEvaluator
from image_loader import ImageLoader
from log_util import Logger
from typing import Optional, Tuple, Dict, Any
from utils.image_utils import ImageUtils
import logging

class PortraitQualityEvaluator:
    """
    ポートレート画像の顔検出、シャープネス、ピンボケ、およびコントラストの評価を行うクラス。
    各評価ロジックは独立したクラスに委譲されています。
    """

    def __init__(self, image_path_or_array: str | np.ndarray, is_raw: bool = False, logger: Optional[Logger] = None):
        """
        画像のパスまたはnumpy配列を受け取り、評価を行います。

        :param image_path_or_array: 画像ファイルのパスまたはnumpy配列
        :param is_raw: RAW画像かどうか
        :param logger: ログを記録するLoggerオブジェクト（省略可能）
        """
        self.is_raw = is_raw
        self.logger = logger or Logger(logger_name='PortraitQualityEvaluator')
        self.image_loader = ImageLoader(logger=self.logger)

        if isinstance(image_path_or_array, str):
            self.image_path = image_path_or_array
            self.file_name = os.path.basename(image_path_or_array)
            self.rgb_image = self.image_loader.load_image(image_path_or_array, output_bps=16 if is_raw else 8)
            self.logger.info(f"画像ファイル {self.file_name} をロードしました")
        elif isinstance(image_path_or_array, np.ndarray):
            self.rgb_image = image_path_or_array
            self.image_path = None
            self.file_name = "numpy配列"
        else:
            raise ValueError("Invalid input type for image data")

    def evaluate(self) -> Dict[str, Any]:
        """
        画像内で顔検出、シャープネス、ピンボケ、コントラストの評価を行います。

        :return: 評価結果の辞書
        """
        self.logger.info(f"評価開始: 画像ファイル {self.file_name}")
        results = {}

        try:
            # 顔検出
            face_result, face_detected, face_region = self._evaluate_face()

            results['face_evaluation'] = face_result
            results['face_detected'] = face_detected

            if face_detected and face_region is not None:
                # 顔領域のシャープネスとコントラスト評価
                face_attribute_results = self._evaluate_face_attributes(face_region)
                results.update({
                    'face_sharpness_score': face_attribute_results['face_sharpness_evaluation'].get('sharpness_score'),
                    'face_contrast_score': face_attribute_results['face_contrast_evaluation'].get('contrast_score'),
                })

            else:
                self.logger.error("顔が検出されなかったか、顔領域が無効です")

            # 画像全体の評価（リサイズ）
            resized_image = ImageUtils.resize_image(self.rgb_image, max_dimension=2048)

            # 各評価メソッドを呼び出し
            results.update({
                'sharpness_score': self._evaluate_sharpness(resized_image).get('sharpness_score'),
                'blurriness_score': self._evaluate_blurriness(resized_image).get('blurriness_score'),
                'contrast_score': self._evaluate_contrast(resized_image).get('contrast_score'),
            })

            return results

        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _evaluate_face(self) -> Tuple[Dict, bool, Optional[np.ndarray]]:
        """
        顔の検出を行います。顔が検出されなかった場合、デフォルトの結果を返します。

        :return: (顔の評価結果, 顔検出フラグ, 検出された顔領域)
        """
        face_image = ImageUtils.resize_image(self.rgb_image, max_dimension=1024)  # 顔検出用にリサイズ
        face_evaluation = self._safe_evaluate(FaceEvaluator, face_image, "顔")

        face_detected = face_evaluation.get('success', False)
        face_region = None

        if face_detected and isinstance(face_evaluation.get('faces', []), list):
            face_data = face_evaluation['faces']
            if face_data:
                box = face_data[0].get('box')
                if box and len(box) == 4:
                    original_bbox = self.calculate_original_bbox(box, face_image.shape, self.rgb_image.shape)
                    face_region = self.rgb_image[original_bbox[1]:original_bbox[1] + original_bbox[3],
                                                  original_bbox[0]:original_bbox[0] + original_bbox[2]]
                else:
                    self.logger.error("顔のバウンディングボックスが無効です")

        return face_evaluation, face_detected, face_region

    def _evaluate_face_attributes(self, face_region: np.ndarray) -> Dict[str, Any]:
        """
        検出された顔領域のシャープネスとコントラストを評価します。

        :param face_region: 検出された顔領域
        :return: 評価結果の辞書
        """
        # 長辺を256ピクセルにリサイズ
        resized_face_region = ImageUtils.resize_image(face_region, max_dimension=256)
        return {
            'face_sharpness_evaluation': self._evaluate_sharpness(resized_face_region),
            'face_contrast_evaluation': self._evaluate_contrast(resized_face_region),
        }

    def calculate_original_bbox(self, box: Tuple[int, int, int, int], face_image_shape: Tuple[int, int, int], original_image_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        顔領域のバウンディングボックスを元の画像のサイズに再計算します。

        :param box: 顔領域のバウンディングボックス (x, y, width, height)
        :param face_image_shape: 顔検出用にリサイズされた画像のサイズ
        :param original_image_shape: 元の画像のサイズ
        :return: 元の画像のバウンディングボックス (x, y, width, height) または None
        """
        try:
            fx = original_image_shape[1] / face_image_shape[1]
            fy = original_image_shape[0] / face_image_shape[0]
            return int(box[0] * fx), int(box[1] * fy), int(box[2] * fx), int(box[3] * fy)
        except Exception as e:
            self.logger.error(f"バウンディングボックスの計算中にエラーが発生しました: {str(e)}")
            return None

    def _safe_evaluate(self, evaluator_class, image: np.ndarray, evaluation_name: str) -> Dict[str, Any]:
        """
        安全に評価を実行し、例外が発生した場合はエラーメッセージを返します。

        :param evaluator_class: 評価クラス
        :param image: 評価対象の画像
        :param evaluation_name: 評価の名前（ログ用）
        :return: 評価結果の辞書
        """
        try:
            evaluator = evaluator_class()
            return evaluator.evaluate(image)
        except Exception as e:
            self.logger.error(f"{evaluation_name}評価中に例外が発生しました: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': f"{evaluation_name}評価中にエラーが発生しました"}

    def _evaluate_sharpness(self, image: np.ndarray) -> Dict[str, Any]:
        """
        シャープネス評価を行います。

        :param image: 評価対象の画像
        :return: 評価結果の辞書
        """
        return self._safe_evaluate(SharpnessEvaluator, image, "シャープネス")

    def _evaluate_blurriness(self, image: np.ndarray) -> Dict[str, Any]:
        """
        ピンボケ評価を行います。

        :param image: 評価対象の画像
        :return: 評価結果の辞書
        """
        return self._safe_evaluate(BlurrinessEvaluator, image, "ピンボケ")

    def _evaluate_contrast(self, image: np.ndarray) -> Dict[str, Any]:
        """
        コントラスト評価を行います。

        :param image: 評価対象の画像
        :return: 評価結果の辞書
        """
        return self._safe_evaluate(ContrastEvaluator, image, "コントラスト")
