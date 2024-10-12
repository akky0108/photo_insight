import numpy as np
import traceback
from evaluators.face_evaluator import FaceEvaluator
from evaluators.sharpness_evaluator import SharpnessEvaluator
from evaluators.blurriness_evaluator import BlurrinessEvaluator
from evaluators.contrast_evaluator import ContrastEvaluator  # ContrastEvaluatorをインポート
from image_loader import ImageLoader
from log_util import Logger
from typing import Optional
from utils.image_utils import ImageUtils
import logging

class PortraitQualityEvaluator:
    """
    ポートレート画像の顔検出、シャープネス、ピンボケ、およびコントラストの評価を行うクラス。
    各評価ロジックは独立したクラスに委譲されています。
    """

    def __init__(self, image_path_or_array, is_raw=False, logger: Optional[Logger] = None):
        """
        画像のパスまたはnumpy配列を受け取り、評価を行います。

        :param image_path_or_array: 画像ファイルのパスまたはnumpy配列
        :param is_raw: RAW画像かどうか
        :param logger: ログを記録するLoggerオブジェクト（省略可能）
        """
        self.is_raw = is_raw
        self.logger = logger if logger else Logger(logger_name='PortraitQualityEvaluator')
        self.image_loader = ImageLoader(logger=self.logger)

        # 画像がRAWの場合、RGBに変換してロード
        if isinstance(image_path_or_array, str):
            self.rgb_image = self.image_loader.load_image(image_path_or_array, output_bps=16 if is_raw else 8)
        elif isinstance(image_path_or_array, np.ndarray):
            self.rgb_image = image_path_or_array
        else:
            raise ValueError("Invalid input type for image data")

        self.log_info_enabled = self.logger.isEnabledFor(logging.INFO)

    def evaluate(self):
        """
        画像内で顔検出、シャープネス、ピンボケ、コントラストの評価を行います。
        """
        results = {}
        try:
            # 顔検出
            face_evaluation, face_detected = self._evaluate_face()
            results.update({
                'face_evaluation': face_evaluation if face_detected else {'message': 'No face detected'},
                'face_detected': face_detected
            })

            # シャープネス評価
            sharpness_evaluation = self._evaluate_sharpness()
            results.update({'sharpness_evaluation': sharpness_evaluation})

            # ピンボケ評価
            blurriness_evaluation = self._evaluate_blurriness()
            results.update({'blurriness_evaluation': blurriness_evaluation})

            # コントラスト評価
            contrast_evaluation = self._evaluate_contrast()
            results.update({'contrast_evaluation': contrast_evaluation})

            return results
        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _evaluate_face(self):
        """
        顔の検出を行います。顔が検出されなかった場合、デフォルトの結果を返します。
        """
        face_image = ImageUtils.resize_image(self.rgb_image, max_dimension=1024)  # 顔検出用にリサイズ
        face_evaluation, face_detected = self._safe_evaluate(FaceEvaluator, face_image, "顔")
        
        if not face_detected:
            face_evaluation = {'message': 'No face detected'}
        
        return face_evaluation, face_detected

    def _evaluate_sharpness(self):
        """
        画像全体のシャープネスを評価し、結果を返します。
        """
        sharpness_image = ImageUtils.resize_image(self.rgb_image, max_dimension=2000)  # シャープネス用リサイズ
        sharpness_score, sharpness_success = self._safe_evaluate(SharpnessEvaluator, sharpness_image, "シャープネス")
        return sharpness_score

    def _evaluate_blurriness(self):
        """
        画像全体のピンボケ度を評価します。
        """
        blurriness_image = self.rgb_image  # ピンボケ評価には元の解像度を使用
        blurriness_score, blurriness_success = self._safe_evaluate(BlurrinessEvaluator, blurriness_image, "ピンボケ")
        return blurriness_score

    def _evaluate_contrast(self):
        """
        画像全体のコントラストを評価します。
        """
        contrast_image = ImageUtils.resize_image(self.rgb_image, max_dimension=2000)  # コントラスト用リサイズ
        contrast_score, contrast_success = self._safe_evaluate(ContrastEvaluator, contrast_image, "コントラスト")
        return contrast_score

    def _safe_evaluate(self, evaluator_class, image, evaluation_name):
        """
        エラーハンドリングを共通化した評価処理。

        :param evaluator_class: 使用する評価クラス
        :param image: 評価対象の画像
        :param evaluation_name: 評価名（ログ出力用）
        :return: 評価結果と評価成功フラグ
        """
        try:
            score = evaluator_class().evaluate(image)
            if self.log_info_enabled:
                self.logger.info(f"{evaluation_name}評価: {score}")
            return score, True
        except Exception as e:
            self.logger.error(f"{evaluation_name}評価エラー: {str(e)}")
            return {'error': f'{evaluation_name} evaluation failed'}, False

