import numpy as np
import os
import traceback

from evaluators.face_evaluator import FaceEvaluator
from evaluators.sharpness_evaluator import SharpnessEvaluator
from evaluators.blurriness_evaluator import BlurrinessEvaluator
from evaluators.contrast_evaluator import ContrastEvaluator
from evaluators.noise_evaluator import NoiseEvaluator  # 修正版 NoiseEvaluator
from evaluators.local_sharpness_evaluator import LocalSharpnessEvaluator
from evaluators.local_contrast_evaluator import LocalContrastEvaluator

from image_loader import ImageLoader
from log_util import Logger
from typing import Optional, Tuple, Dict, Any
from utils.image_utils import ImageUtils

class PortraitQualityEvaluator:
    """
    ポートレート画像の顔検出、シャープネス、ピンボケ、コントラスト、ノイズの評価を行うクラス。
    各評価ロジックは独立したクラスに委譲されています。
    """

    def __init__(self, image_path_or_array: str | np.ndarray, is_raw: bool = False, logger: Optional[Logger] = None, file_name: Optional[str] = None, max_noise_value: float = 100.0, local_region_size: int = 32):
        """
        画像のパスまたはnumpy配列を受け取り、評価を行います。

        :param image_path_or_array: 画像ファイルのパスまたはnumpy配列
        :param is_raw: RAW画像かどうか
        :param logger: ログを記録するLoggerオブジェクト（省略可能）
        :param logger: ファイル名（省略可能）
        :param max_noise_value: ノイズ評価時の最大ノイズ閾値
        :param local_region_size: 局所シャープネス・コントラスト評価の領域サイズ
        """
        self.is_raw = is_raw
        self.logger = logger or Logger(logger_name='PortraitQualityEvaluator')
        self.image_loader = ImageLoader(logger=self.logger)
        self.noise_evaluator = NoiseEvaluator(max_noise_value=max_noise_value)
        self.local_sharpness_evaluator = LocalSharpnessEvaluator(block_size=local_region_size)
        self.local_contrast_evaluator = LocalContrastEvaluator(block_size=local_region_size)

        if isinstance(image_path_or_array, str):
            self.image_path = image_path_or_array
            self.file_name = os.path.basename(image_path_or_array)
            self.rgb_image = self.image_loader.load_image(image_path_or_array, output_bps=16 if is_raw else 8)
        elif isinstance(image_path_or_array, np.ndarray):
            self.rgb_image = image_path_or_array
            self.image_path = None
            self.file_name = file_name if file_name else "numpy配列"
        else:
            raise ValueError("無効な入力タイプの画像データ")

        self.logger.info(f"画像ファイル {self.file_name} をロードしました")

    def evaluate(self) -> Dict[str, Any]:
        """
        画像内で顔検出、シャープネス、ピンボケ、コントラスト、ノイズの評価を行います。

        :return: 評価結果の辞書
        """
        self.logger.info(f"評価開始: 画像ファイル {self.file_name}")
        results = {}

        try:
            # 顔検出と評価
            face_result, face_detected, face_region = self._evaluate_face()
            results['face_evaluation'] = face_result
            results['face_detected'] = face_detected

            # 顔検出成功時、顔領域の詳細評価
            if face_detected and face_region is not None:
                face_attribute_results = self._evaluate_face_attributes(face_region)
                results.update({
                    'face_sharpness_score': face_attribute_results['face_sharpness_evaluation'].get('sharpness_score'),
                    'face_contrast_score': face_attribute_results['face_contrast_evaluation'].get('contrast_score'),
                    'face_noise_score': face_attribute_results['face_noise_evaluation'].get('noise_score'),  # 顔のノイズ評価
                    'face_local_sharpness_score': face_attribute_results.get('face_local_sharpness_score', 0.0),
                    'face_local_sharpness_std': face_attribute_results.get('face_local_sharpness_std', 0.0),
                    'face_local_contrast_score': face_attribute_results.get('face_local_contrast_score', 0.0),
                    'face_local_contrast_std': face_attribute_results.get('face_local_contrast_std', 0.0),                })
            else:
                self.logger.error("顔が検出されなかったか、顔領域が無効です")

            # 画像全体のリサイズと評価
            resized_image = ImageUtils.resize_image(self.rgb_image, max_dimension=2048)
            if resized_image is None:
                self.logger.error("resized_image が None です。評価をスキップします。")
                return {}

            # 各評価関数の戻り値を取得し、Noneならエラーログを出力
            sharpness_eval = self._evaluate_sharpness(resized_image)
            if sharpness_eval is None:
                self.logger.error("sharpness_eval が None です: %s", self.image_path)
                sharpness_eval = {}
            else:
                self.logger.info("sharpness_eval 結果: %s", sharpness_eval)

            blurriness_eval = self._evaluate_blurriness(resized_image)
            if blurriness_eval is None:
                self.logger.error("blurriness_eval が None です: %s", self.image_path)
                blurriness_eval = {}
            else:
                self.logger.info("blurriness_eval 結果: %s", blurriness_eval)

            contrast_eval = self._evaluate_contrast(resized_image)
            if contrast_eval is None:
                self.logger.error("contrast_eval が None です: %s", self.image_path)
                contrast_eval = {}
            else:
                self.logger.info("contrast_eval 結果: %s", contrast_eval)

            noise_eval = self._evaluate_noise(resized_image)
            if noise_eval is None:
                self.logger.error("noise_eval が None です: %s", self.image_path)
                noise_eval = {}
            else:
                self.logger.info("noise_eval 結果: %s", noise_eval)

            local_sharpness_eval = self._evaluate_local_sharpness(resized_image)
            if local_sharpness_eval is None:
                self.logger.error("local_sharpness_eval が None です: %s", self.image_path)
                local_sharpness_eval = {}
            else:
                self.logger.info("local_sharpness_eval 結果: %s", local_sharpness_eval)

            local_contrast_eval = self._evaluate_local_contrast(resized_image)
            if local_contrast_eval is None:
                self.logger.error("local_contrast_eval が None です: %s", self.image_path)
                local_contrast_eval = {}
            else:
                self.logger.info("local_contrast_eval 結果: %s", local_contrast_eval)

            # 結果を辞書に格納（None回避のためデフォルト値を設定）
            results.update({
                'sharpness_score': sharpness_eval.get('sharpness_score', 0.0),
                'blurriness_score': blurriness_eval.get('blurriness_score', 0.0),
                'contrast_score': contrast_eval.get('contrast_score', 0.0),
                'noise_score': noise_eval.get('noise_score', 0.0),
                'local_sharpness_score': local_sharpness_eval.get('local_sharpness_score', 0.0),
                'local_sharpness_std': local_sharpness_eval.get('local_sharpness_std', 0.0),
                'local_contrast_score': local_contrast_eval.get('local_contrast_score', 0.0),
                'local_contrast_std': local_contrast_eval.get('local_contrast_std', 0.0),
            })

            return results  # 評価結果を返す

        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}  # エラーが発生した場合、空の辞書を返す

    def _evaluate_face(self) -> Tuple[Dict, bool, Optional[np.ndarray]]:
        """
        顔検出を行います。顔が検出されなかった場合、デフォルトの結果を返します。

        :return: (顔の評価結果, 顔検出フラグ, 検出された顔領域)
        """
        face_image = ImageUtils.resize_image(self.rgb_image, max_dimension=1024)
        face_evaluation = self._safe_evaluate(FaceEvaluator, face_image, "顔")

        face_detected = face_evaluation.get('success', False)
        face_region = None

        if face_detected:
            face_data = face_evaluation['faces']
            if face_data:
                box = face_data[0].get('box')
                if box and len(box) == 4:
                    original_bbox = self.calculate_original_bbox(box, face_image.shape, self.rgb_image.shape)
                    face_region = self.rgb_image[original_bbox[1]:original_bbox[1] + original_bbox[3],
                                                  original_bbox[0]:original_bbox[0] + original_bbox[2]]
        return face_evaluation, face_detected, face_region

    def _evaluate_face_attributes(self, face_region: np.ndarray) -> Dict[str, Any]:
        """
        検出された顔領域のシャープネス、コントラスト、およびノイズを評価します。

        :param face_region: 検出された顔領域
        :return: 評価結果の辞書
        """
        resized_face_region = ImageUtils.resize_image(face_region, max_dimension=256)

        if resized_face_region is None:
            self.logger.error("resized_face_region が None です。顔評価をスキップします。")
            return {}

        # `_evaluate_local_sharpness()` と `_evaluate_local_contrast()` の結果を1回だけ計算
        local_sharpness = self._evaluate_local_sharpness(resized_face_region)
        if local_sharpness is None:
            self.logger.error("local_sharpness が None です: %s", self.image_path)
            local_sharpness = {}
        else:
            self.logger.info("local_sharpness 結果: %s", local_sharpness)

        local_contrast = self._evaluate_local_contrast(resized_face_region)
        if local_contrast is None:
            self.logger.error("local_contrast が None です: %s", self.image_path)
            local_contrast = {}
        else:
            self.logger.info("local_contrast 結果: %s", local_contrast)

        sharpness_eval = self._evaluate_sharpness(resized_face_region)
        if sharpness_eval is None:
            self.logger.error("sharpness_eval が None です: %s", self.image_path)
        else:
            self.logger.info("sharpness_eval 結果: %s", sharpness_eval)

        contrast_eval = self._evaluate_contrast(resized_face_region)
        if contrast_eval is None:
            self.logger.error("contrast_eval が None です: %s", self.image_path)
        else:
            self.logger.info("contrast_eval 結果: %s", contrast_eval)

        noise_eval = self._evaluate_noise(resized_face_region)
        if noise_eval is None:
            self.logger.error("noise_eval が None です: %s", self.image_path)
        else:
            self.logger.info("noise_eval 結果: %s", noise_eval)

        return {
            'face_sharpness_evaluation': sharpness_eval if sharpness_eval else {},
            'face_contrast_evaluation': contrast_eval if contrast_eval else {},
            'face_noise_evaluation': noise_eval if noise_eval else {},  # 顔のノイズ評価
            'face_local_sharpness_score': local_sharpness.get('local_sharpness_score', 0.0),
            'face_local_sharpness_std': local_sharpness.get('local_sharpness_std', 0.0),
            'face_local_contrast_score': local_contrast.get('local_contrast_score', 0.0),
            'face_local_contrast_std': local_contrast.get('local_contrast_std', 0.0),
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
            # NoiseEvaluator, LocalSharpnessEvaluator, LocalContrastEvaluator は事前にインスタンス化
            if evaluator_class == NoiseEvaluator:
                evaluator = self.noise_evaluator
            elif evaluator_class == LocalSharpnessEvaluator:
                evaluator = self.local_sharpness_evaluator
            elif evaluator_class == LocalContrastEvaluator:
                evaluator = self.local_contrast_evaluator
            else:
                evaluator = evaluator_class()

            return evaluator.evaluate(image)
        except Exception as e:
            self.logger.error(f"{evaluation_name}評価中に例外が発生しました: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {'error': f"{evaluation_name}評価中にエラーが発生しました"}

    # 各評価関数
    def _evaluate_sharpness(self, image: np.ndarray) -> Dict[str, Any]:
        return self._safe_evaluate(SharpnessEvaluator, image, "シャープネス")

    def _evaluate_blurriness(self, image: np.ndarray) -> Dict[str, Any]:
        return self._safe_evaluate(BlurrinessEvaluator, image, "ピンボケ")

    def _evaluate_contrast(self, image: np.ndarray) -> Dict[str, Any]:
        return self._safe_evaluate(ContrastEvaluator, image, "コントラスト")

    def _evaluate_noise(self, image: np.ndarray) -> Dict[str, Any]:
        return self._safe_evaluate(NoiseEvaluator, image, "ノイズ")  # ノイズ評価

    def _evaluate_local_sharpness(self, image: np.ndarray) -> Dict[str, Any]:
        return self._safe_evaluate(LocalSharpnessEvaluator, image, "局所シャープネス")

    def _evaluate_local_contrast(self, image: np.ndarray) -> Dict[str, Any]:
        return self._safe_evaluate(LocalContrastEvaluator, image, "局所コントラスト")

    def evaluate_composition(self, image):
        """
        ポートレート全般の構図評価を行う。
        - 顔主体（バストアップ）
        - 半身（ウエストアップ）
        - 全身（フルボディ）
        - 動きのあるポーズ
        """
        score = 0
        composition_type = self.determine_composition_type(image)
        
        if composition_type == "bust_up":
            score += self.evaluate_face_position(image) * 1.2  # バストアップ時の顔の重要性を強調
        elif composition_type == "waist_up":
            score += self.evaluate_face_position(image) * 1.0
            score += self.evaluate_body_proportion(image) * 1.0
        elif composition_type == "full_body":
            score += self.evaluate_face_position(image) * 0.8
            score += self.evaluate_body_proportion(image) * 1.5  # 全身では身体の比率がより重要
            score += self.evaluate_pose_dynamics(image) * 1.2  # 動きのあるポーズの評価を追加
        
        score += self.evaluate_balance(image)  # フレーム内バランス評価
        score += self.evaluate_background(image)  # 背景との関係評価
        return score

    def determine_composition_type(self, image):
        """ 画像から構図タイプ（バストアップ、ウエストアップ、フルボディ）を判定 """
        # 仮の処理: 画像サイズや顔の検出範囲を基に判定する
        return "full_body"  # 仮の値

    def evaluate_face_position(self, image):
        """顔の位置が適切か評価（ルールオブサード、中央配置など）"""
        return 1  # 仮のスコア

    def evaluate_balance(self, image):
        """全体のバランス評価（余白の適切さ、トリミングミスなど）"""
        return 1  # 仮のスコア

    def evaluate_background(self, image):
        """背景との分離が適切か評価（不要なオブジェクトの影響）"""
        return 1  # 仮のスコア

    def evaluate_body_proportion(self, image):
        """手足のトリミングミスや歪みの評価"""
        return 1  # 仮のスコア

    def evaluate_pose_dynamics(self, image):
        """ポーズのダイナミクスを評価（静的・動的）"""
        return 1  # 仮のスコア
