import numpy as np
import os
import traceback
import cv2

from typing import Optional, Tuple, Dict, Any

from evaluators.face_evaluator import FaceEvaluator
from evaluators.sharpness_evaluator import SharpnessEvaluator
from evaluators.blurriness_evaluator import BlurrinessEvaluator
from evaluators.contrast_evaluator import ContrastEvaluator
from evaluators.noise_evaluator import NoiseEvaluator
from evaluators.local_sharpness_evaluator import LocalSharpnessEvaluator
from evaluators.local_contrast_evaluator import LocalContrastEvaluator

from detectors.body_detection import FullBodyDetector  # 構図評価用

from image_loader import ImageLoader
from log_util import Logger
from utils.image_utils import ImageUtils

class PortraitQualityEvaluator:
    """
    ポートレート画像の顔検出、シャープネス、ピンボケ、コントラスト、ノイズの評価を行うクラス。
    各評価ロジックは独立したクラスに委譲されています。
    """

    def __init__(self, image_input: str | np.ndarray, is_raw: bool = False, logger: Optional[Logger] = None, 
                 file_name: Optional[str] = None, max_noise_value: float = 100.0, local_region_size: int = 32):
        """
        PortraitQualityEvaluator クラスのコンストラクタ。
        
        :param image_input: 画像の入力（ファイルパスまたは NumPy 配列）
        :param is_raw: RAW 画像かどうか（デフォルト: False）
        :param logger: ログ出力用の Logger インスタンス（省略時はデフォルトロガーを使用）
        :param file_name: 画像ファイル名（省略時は入力から自動取得）
        :param max_noise_value: ノイズ評価の最大値（デフォルト: 100.0）
        :param local_region_size: ローカル評価時のブロックサイズ（デフォルト: 32）
        """
        self.is_raw = is_raw
        self.logger = logger or Logger(logger_name='PortraitQualityEvaluator')
        self.image_loader = ImageLoader(logger=self.logger)
        self.file_name = file_name if isinstance(image_input, np.ndarray) else os.path.basename(image_input)
        self.image_path = image_input if isinstance(image_input, str) else None
        
        self.rgb_image = self._load_image(image_input)
        self.evaluators = self._initialize_evaluators(max_noise_value, local_region_size)
        self.logger.info(f"画像ファイル {self.file_name} をロードしました")

    def _load_image(self, image_input: str | np.ndarray) -> np.ndarray:
        """
        画像をロードするメソッド。
        
        :param image_input: 画像の入力（ファイルパスまたは NumPy 配列）
        :return: ロードした画像の NumPy 配列
        """
        if isinstance(image_input, str):
            return self.image_loader.load_image(image_input, output_bps=16 if self.is_raw else 8)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("無効な入力タイプの画像データ")
    
    def _initialize_evaluators(self, max_noise_value: float, local_region_size: int) -> Dict[str, Any]:
        """
        各種評価モジュールを初期化する。
        
        :param max_noise_value: ノイズ評価の最大値
        :param local_region_size: ローカル評価時のブロックサイズ
        :return: 初期化された評価器の辞書
        """
        return {
            "sharpness": SharpnessEvaluator(),
            "blurriness": BlurrinessEvaluator(),
            "contrast": ContrastEvaluator(),
            "noise": NoiseEvaluator(max_noise_value=max_noise_value),
            "local_sharpness": LocalSharpnessEvaluator(block_size=local_region_size),
            "local_contrast": LocalContrastEvaluator(block_size=local_region_size),
            "face": FaceEvaluator(backend='insightface')
        }

    def evaluate(self) -> Dict[str, Any]:
        """
        画像の品質を評価するメソッド。
        
        :return: 各評価指標のスコアを含む辞書
        """
        self.logger.info(f"評価開始: 画像ファイル {self.file_name}")
        results = {}

        try:
            # 画像を2048pxにリサイズして評価用に使用
            resized_image = ImageUtils.resize_image(self.rgb_image, max_dimension=2048)
            if resized_image is None:
                self.logger.error("resized_image が None です。評価をスキップします。")
                return {}

            face_detected = False  # デフォルト値
            faces = {}
            face_boxes = None  # 顔の領域情報

            for key, evaluator in self.evaluators.items():
                # 顔評価の場合は1024pxにリサイズした画像を使用
                target_image = ImageUtils.resize_image(self.rgb_image, max_dimension=1024) if key == "face" else resized_image

                if key == "face":
                    face_result = self._evaluate_face(evaluator, self.rgb_image)  # 修正
                    results.update(face_result)
                else:
                    results.update(self._safe_evaluate(evaluator, target_image, key))

            return results
        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
        
    def _evaluate_face(self, evaluator, image):
        """
        顔評価を高速化しつつ高精度を維持する処理:
        1. 1024px にリサイズして大まかな顔検出
        2. 検出した顔の領域をクロップし、元解像度で再検出
        """

        # 元画像の解像度を取得
        original_h, original_w = image.shape[:2]
        
        # 低解像度(1024px)にリサイズ
        resized_image = ImageUtils.resize_image(image, max_dimension=1024)

        # 低解像度画像で顔検出
        face_result = evaluator.evaluate(resized_image)
        face_detected = face_result.get("face_detected", False)

        if not face_detected:
            print("顔が検出されませんでした。")
            return face_result  # 検出できなかった場合はそのまま結果を返す

        # 検出された顔情報を取得
        faces = face_result.get("faces", [])
        face_boxes = [face["box"] for face in faces]  # 顔の位置情報を取得
        
        print("低解像度で検出された顔の座標:", face_boxes)

        # 高解像度での再検出用リスト
        refined_faces = []
        scale_factor = original_w / 1024  # スケール倍率を計算

        for face in faces:
            x, y, w, h = face["box"]

            # 座標を元の解像度にスケールバック
            x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)

            # 顔周辺に余白を追加（顔サイズの 10%）
            padding = int(min(w, h) * 0.2)
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(original_w, x + w + padding), min(original_h, y + h + padding)

            # 高解像度画像から顔領域をクロップ
            cropped_face = image[y1:y2, x1:x2]

            cv2.imwrite(f"debug_crop_{x}_{y}.jpg", cropped_face)


            # 高解像度で顔を再検出
            high_res_result = evaluator.evaluate(cropped_face)
            high_res_faces = high_res_result.get("faces", [])

            for hf in high_res_faces:
                hx, hy, hw, hh = hf["box"]
                
                # 高解像度での検出結果を元画像の座標系に変換
                refined_face_box = (hx + x1, hy + y1, hw, hh)
                refined_faces.append({
                    "box": refined_face_box,
                    "confidence": hf["confidence"],
                    "landmarks": hf["landmarks"]
                })

        print("高解像度で再検出された顔の座標:", [face["box"] for face in refined_faces])

        # 最終的な顔情報を更新して返す
        face_result["faces"] = refined_faces
        return face_result

    def _upscale_image(self, image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

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

    def _get_value(self, result: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        指定されたキーの値を取得する。キーに "local_" が含まれる場合は "_std" の値も取得する。

        :param result: 評価結果の辞書
        :param key: 取得したいキー（例: 'sharpness_score', 'success'）
        :param default: キーが存在しない場合のデフォルト値（デフォルトはNone）
        :return: 指定キーの値。キーに "local_" が含まれる場合は "_std" の値も辞書に含める
        """

        return result.get(key, default)       

    def _safe_evaluate(self, evaluator, image: np.ndarray, name: str) -> Dict[str, Any]:
        """
        安全に評価を実行し、例外が発生した場合はエラーメッセージを返します。

        :param evaluator: 評価クラスのインスタンス
        :param image: 評価対象の画像
        :param name: 評価の名前（ログ用）
        :return: 評価結果の辞書
        """
        try:
            result = evaluator.evaluate(image)
            extracted_results = {}

            # 基本スコアの取得
            score_key = f"{name}_score"
            extracted_results[score_key] = self._get_value(result, score_key, None)

            # "local_" を含む場合は "_std" の値も取得
            if "local_" in name:
                std_key = f"{name}_std"
                extracted_results[std_key] = self._get_value(result, std_key, None)

            return extracted_results
        except Exception as e:
            self.logger.error(f"{name} 評価中にエラーが発生しました ({type(e).__name__}): {str(e)}")
            self.logger.error(traceback.format_exc())
            return {f"{name}_score": None}

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
