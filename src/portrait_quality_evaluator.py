import numpy as np
import traceback
from evaluators.sharpness_evaluator import SharpnessEvaluator
from evaluators.contrast_evaluator import ContrastEvaluator
from evaluators.noise_evaluator import NoiseEvaluator
from evaluators.wavelet_sharpness_evaluator import WaveletSharpnessEvaluator
from evaluators.blurriness_evaluator import BlurrinessEvaluator
from evaluators.face_evaluator import FaceEvaluator
from image_loader import ImageLoader
from log_util import Logger
from utils.math_utils import MathUtils
from typing import Optional
import cv2
import logging

class PortraitQualityEvaluator:
    """
    ポートレート画像の品質を評価するクラス。
    画像の品質指標としてシャープネス、コントラスト、ノイズ、ウェーブレットシャープネス、ぼやけ、顔評価を使用。
    """

    def __init__(self, image_path_or_array, is_raw=False, weights=None, thresholds=None, logger: Optional[Logger] = None):
        """
        コンストラクタ。画像のパスまたはnumpy配列を指定して、品質評価を行います。

        :param image_path_or_array: 画像ファイルのパスまたはnumpy配列
        :param is_raw: RAW画像かどうか（デフォルトはFalse）
        :param weights: 各評価項目の重み（デフォルトはNoneで、自動設定）
        :param thresholds: シャープネス評価の閾値（デフォルトはNoneで、自動設定）
        :param logger: ログ出力に使用するLoggerオブジェクト（省略可能）
        """
        self.is_raw = is_raw
        self.logger = logger if logger else Logger(logger_name='PortraitQualityEvaluator')
        self.image_loader = ImageLoader(logger=self.logger)

        # 画像がRAWの場合、RGBに変換してロード
        if is_raw and isinstance(image_path_or_array, str):
            self.rgb_image = self.image_loader.load_image(image_path_or_array, output_bps=8)
        else:
            self.rgb_image = np.array(image_path_or_array)

        # 重みと閾値が指定されていない場合はデフォルト値を使用
        self.weights = weights if weights else self._default_weights()
        self.thresholds = thresholds if thresholds else self._default_thresholds()
        
        # 評価器の遅延初期化のため、空の辞書を作成
        self.evaluators = {}
        
        # キャッシュされたリサイズ画像
        self.resized_image = None
        
        # ログレベルのチェックを事前に行う
        self.log_info_enabled = self.logger.isEnabledFor(logging.INFO)

    def _default_weights(self):
        """デフォルトの評価項目の重みを返す"""
        return {
            'sharpness': 0.2,
            'contrast': 0.2,
            'noise': 0.2,
            'wavelet_sharpness': 0.1,
            'face': 0.2,
            'blurriness': 0.1
        }

    def _default_thresholds(self):
        """デフォルトの閾値を返す"""
        return {
            'sharpness': {'good': 0.8, 'average': 0.5},
            'blurriness': {'good': 0.2, 'average': 0.5}
        }

    def _get_evaluator(self, metric_name):
        """
        指定した評価項目の評価器を遅延初期化する。
        まだ初期化されていない場合のみ、新しくインスタンス化する。
        """
        if metric_name not in self.evaluators:
            evaluator_class = {
                'sharpness': SharpnessEvaluator,
                'contrast': ContrastEvaluator,
                'noise': NoiseEvaluator,
                'wavelet_sharpness': WaveletSharpnessEvaluator,
                'blurriness': BlurrinessEvaluator,
                'face': FaceEvaluator
            }.get(metric_name)

            # 特別な処理はインスタンス化後に
            if metric_name == 'face':
                self.evaluators[metric_name] = evaluator_class(weights=self.weights)
            else:
                self.evaluators[metric_name] = evaluator_class()

        return self.evaluators[metric_name]

    def evaluate(self):
        """
        画像全体の評価を行い、結果を返す。
        顔検出が成功しなければ、画像全体で評価を行う。
        """
        results = {}
        try:
            # 顔評価の辞書を取得
            face_evaluation, face_weight, face_detected = self._evaluate_face()

            # メトリックの評価は共通化
            metric_results = {metric: self._evaluate_metric_as_dict(metric) for metric in 
                              ['sharpness', 'contrast', 'noise', 'wavelet_sharpness', 'blurriness']}
            results.update(metric_results)

            # 総合スコアの計算（顔検出の結果によって分岐）
            overall_score = self._calculate_overall_score(metric_results, face_evaluation, face_weight, face_detected)
            results.update({
                'overall_score': overall_score,
                'face_evaluation': face_evaluation if face_detected else {},
                'face_detected': face_detected
            })

            return results
        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def _evaluate_metric_as_dict(self, metric_name):
        """
        指定した評価項目のスコアを辞書形式で返す。
        ログ出力が有効な場合にのみ、評価結果をログに出力する。
        """
        try:
            evaluator = self._get_evaluator(metric_name)
            score = evaluator.evaluate(self.rgb_image)

            if self.log_info_enabled:
                self.logger.info(f"{metric_name.capitalize()} score: {score}")

            return {'score': score, 'additional_info': {}}
        except Exception as e:
            self.logger.error(f"{metric_name}評価エラー: {str(e)}")
            return {'score': 0, 'additional_info': {}}

    def _evaluate_face(self):
        """
        顔の検出と評価を行う。
        リサイズした画像をキャッシュし、顔検出が失敗した場合には早期リターンする。
        """
        try:
            # リサイズ画像がキャッシュされていなければリサイズ処理を行う
            if self.resized_image is None:
                self.resized_image = self._resize_image(self.rgb_image, max_dimension=512)

            # 顔評価を実行
            face_evaluation, face_weight, face_detected = self._get_evaluator('face').detect_and_evaluate(self.resized_image, is_raw=self.is_raw)

            if self.log_info_enabled:
                self.logger.info(f"顔評価: {face_evaluation}, 顔検出: {face_detected}")

            return face_evaluation, face_weight, face_detected
        except Exception as e:
            self.logger.error(f"顔評価エラー: {str(e)}")
            return {}, 0, False

    def _resize_image(self, image, max_dimension):
        """
        画像をアスペクト比を維持しながらリサイズする。
        指定された最大サイズに基づいて画像を縮小する。
        """
        h, w, _ = image.shape
        if max(h, w) <= max_dimension:
            return image
        scale = max_dimension / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)))

    def _calculate_overall_score(self, metric_results, face_eval, face_weight, face_detected):
        """
        総合スコアを計算し、顔の評価も考慮する。
        顔が検出されなかった場合、顔の重みを他の評価項目の重みで補う。
        """
        try:
            adjusted_weights = self.weights.copy()
            if not face_detected:
                # 顔の重みを他の評価項目に再分配
                face_weight = 0
                non_face_weight_sum = sum(adjusted_weights[metric] for metric in adjusted_weights if metric != 'face')

                for metric in adjusted_weights:
                    if metric != 'face':
                        adjusted_weights[metric] += self.weights['face'] * (adjusted_weights[metric] / non_face_weight_sum)

            # 評価項目の重み付きスコアを計算
            weighted_scores = sum(
                metric_results[metric]['score'] * adjusted_weights[metric]
                for metric in ['sharpness', 'contrast', 'noise', 'wavelet_sharpness', 'blurriness']
            )

            # 顔評価がある場合は、顔の重みも考慮
            if face_detected:
                weighted_scores += face_weight * sum(face_eval.values())

            return round(weighted_scores, 2)
        except Exception as e:
            self.logger.error(f"総合スコア計算エラー: {str(e)}")
            return 0
