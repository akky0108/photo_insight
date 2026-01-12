import cv2
import gc
import numpy as np
import os
import traceback

from typing import Optional, Tuple, Dict, Any

# 自作ライブラリ（変更不可）
from evaluators.face_evaluator import FaceEvaluator
from evaluators.sharpness_evaluator import SharpnessEvaluator
from evaluators.blurriness_evaluator import BlurrinessEvaluator
from evaluators.contrast_evaluator import ContrastEvaluator
from evaluators.composite_composition_evaluator import CompositeCompositionEvaluator
from evaluators.noise_evaluator import NoiseEvaluator
from evaluators.exposure_evaluator import ExposureEvaluator
from evaluators.local_sharpness_evaluator import LocalSharpnessEvaluator
from evaluators.local_contrast_evaluator import LocalContrastEvaluator
from evaluators.rule_based_composition_evaluator import RuleBasedCompositionEvaluator
from evaluators.color_balance_evaluator import ColorBalanceEvaluator
from detectors.body_detection import FullBodyDetector
from utils.app_logger import Logger

from image_utils.image_preprocessor import ImagePreprocessor
from face_detectors.face_processor import FaceProcessor

from evaluators.quality_thresholds import QualityThresholds
from evaluators.portrait_accept_rules import decide_accept

class PortraitQualityEvaluator:
    """
    ポートレート画像の品質を多角的に評価するためのクラス。

    顔認識、構図、シャープネス、ノイズ、コントラスト等を統合的にスコア化し、
    画像全体および顔領域に対する評価結果を辞書形式で返却する。

    引数:
        image_input (str | np.ndarray): 評価対象の画像（パスまたはRGB画像データ）
        is_raw (bool): RAW画像として処理するかどうか
        logger (Optional[Logger]): ロガーインスタンス（未指定時は内部生成）
        file_name (Optional[str]): ログ出力時などに使うファイル名
        max_noise_value (float): ノイズ評価の正規化上限値
        local_region_size (int): 局所評価時のブロックサイズ
        preprocessor_resize_size (Tuple[int, int]): 前処理時のリサイズサイズ

    戻り値:
        Dict[str, Any]: 各評価項目のスコアを格納した辞書
    """

    def __init__(
        self,
        image_input: str | np.ndarray,
        is_raw: bool = False,
        logger: Optional[Logger] = None,
        file_name: Optional[str] = None,
        max_noise_value: float = 100.0,
        local_region_size: int = 32,
        preprocessor_resize_size: Tuple[int, int] = (224, 224),
        face_processor: Optional[FaceProcessor] = None,
        skip_face_processing: bool = False,
        config_manager: Optional[Any] = None,        # ★ 追加
        quality_profile: str = "portrait",          # ★ 追加（将来 landscape 等も想定）
        thresholds_path: Optional[str] = None,      # ★ 追加（直指定したい場合）
    ):
        self.is_raw = is_raw
        self.logger = logger or Logger(logger_name="PortraitQualityEvaluator")
        self.config_manager = config_manager
        self.file_name = (
            file_name
            if isinstance(image_input, np.ndarray)
            else os.path.basename(image_input)
        )
        self.image_path = image_input if isinstance(image_input, str) else None
        self.skip_face_processing = skip_face_processing
        self.image = None

        # 前処理器で画像を一括取得
        self.face_evaluator = FaceEvaluator(backend="insightface")
        self.face_processor = FaceProcessor(self.face_evaluator, logger=self.logger)

        self.preprocessor = ImagePreprocessor(
            logger=self.logger, is_raw=self.is_raw, gamma=1.2
        )

        images = self.preprocessor.load_and_resize(image_input)

        # RGB（互換）
        self.rgb_image = images["original"]
        self.resized_image_2048 = images["resized_2048"]
        self.resized_image_1024 = images["resized_1024"]

        # BGR
        self.bgr_image = images["original_bgr"]
        self.resized_image_2048_bgr = images["resized_2048_bgr"]
        self.resized_image_1024_bgr = images["resized_1024_bgr"]

        # ★ 評価用 uint8（ここが恒久対策）
        self.rgb_u8 = images["original_u8"]
        self.resized_2048_bgr_u8 = images["resized_2048_bgr_u8"]
        self.bgr_u8 = images["original_bgr_u8"]

        img = self.resized_2048_bgr_u8
        self.logger.info(
            f"[debug] eval_img dtype={img.dtype}, min={img.min()}, max={img.max()}, mean={img.mean():.2f}"
        )

        self.evaluators = {
            "face": self.face_evaluator,
            "sharpness": SharpnessEvaluator(),
            "blurriness": BlurrinessEvaluator(),
            "contrast": ContrastEvaluator(),
            "noise": NoiseEvaluator(max_noise_value=max_noise_value),
            "local_sharpness": LocalSharpnessEvaluator(block_size=local_region_size),
            "local_contrast": LocalContrastEvaluator(block_size=local_region_size),
            "exposure": ExposureEvaluator(),
            "color_balance": ColorBalanceEvaluator(),
        }

        self.body_detector = FullBodyDetector()

        # self.composition_evaluator = RuleBasedCompositionEvaluator(logger=self.logger)
        # 変更後: 顔＋全身を統合する Composite を使う
        self.composition_evaluator = CompositeCompositionEvaluator(
            logger=self.logger,
            config=None,   # 将来 config を渡したくなったらここに
        )

        self.logger.info(f"画像ファイル {self.file_name} をロードしました")

        # 閾値系の初期化だけ 1 行にまとめる
        self._init_thresholds(quality_profile=quality_profile, thresholds_path=thresholds_path)


    def _init_thresholds(self, quality_profile: str, thresholds_path: Optional[str]) -> None:
        """
        YAML ファイルから評価の閾値テーブルをロードして、
        self.decision_thresholds にフラットな dict で格納する。
        """
        if thresholds_path is None:
            if self.config_manager is not None:
                thresholds_path = self.config_manager.get(
                    "quality_thresholds_path",
                    default="config/quality_thresholds.yaml",
                )
            else:
                thresholds_path = "config/quality_thresholds.yaml"

        self.quality_thresholds = QualityThresholds(path=thresholds_path)

        try:
            self.threshold_profile = self.quality_thresholds.profile(quality_profile)
        except Exception as e:
            self.logger.warning(f"Quality thresholds load failed: {e}")
            self.threshold_profile = {}

        decision = self.threshold_profile.get("decision", {})

        self.decision_thresholds: Dict[str, float] = {
            # ノイズ
            "noise_ok": float(decision.get("noise_ok", 0.5)),
            "noise_good": float(decision.get("noise_good", 0.75)),
            "face_noise_good": float(decision.get("face_noise_good", 0.75)),

            # ★ full body / shot_type 判定用（撮影距離・切れ具合）
            "full_body_body_height_min": float(decision.get("full_body_body_height_min", 0.30)),
            "full_body_cut_risk_max_for_shot_type": float(decision.get("full_body_cut_risk_max_for_shot_type", 0.90)),
            "full_body_footroom_min_for_shot_type": float(decision.get("full_body_footroom_min_for_shot_type", 0.00)),

            # ★ seated（座り）ショット判定用
            "seated_center_y_min": float(decision.get("seated_center_y_min", 0.50)),
            "seated_center_y_max": float(decision.get("seated_center_y_max", 0.75)),
            "seated_footroom_max": float(decision.get("seated_footroom_max", 0.22)),
            "seated_body_height_min": float(decision.get("seated_body_height_min", 0.30)),

            # ★ upper_body / face_only 判定用
            "upper_body_headroom_min": float(decision.get("upper_body_headroom_min", 0.15)),
            "upper_body_footroom_max": float(decision.get("upper_body_footroom_max", 0.15)),
            "face_only_body_height_max": float(decision.get("face_only_body_height_max", 0.35)),
            "center_side_margin_min": float(decision.get("center_side_margin_min", 0.02)),

            # full body（accept ルート用）
            "full_body_face_noise_min": float(decision.get("full_body_face_noise_min", 0.5)),
            "full_body_pose_min_100": float(decision.get("full_body_pose_min_100", 55.0)),
            "full_body_cut_risk_max": float(decision.get("full_body_cut_risk_max", 0.6)),
            "blurriness_min_full_body": float(decision.get("blurriness_min_full_body", 0.45)),
            "exposure_min_common": float(decision.get("exposure_min_common", 0.5)),
            "full_body_footroom_min": float(decision.get("full_body_footroom_min", 0.0)),

            # full body 用コントラスト
            "full_body_contrast_min": float(decision.get("full_body_contrast_min", 0.40)),
            "full_body_face_contrast_min": float(decision.get("full_body_face_contrast_min", 0.50)),

            # composition ルート
            "composition_score_min": float(decision.get("composition_score_min", 0.75)),
            "framing_score_min": float(decision.get("framing_score_min", 0.5)),
            "lead_room_score_min": float(decision.get("lead_room_score_min", 0.10)),

            # face_quality ルート
            "face_quality_exposure_min": float(decision.get("face_quality_exposure_min", 0.5)),
            "face_quality_face_sharpness_min": float(decision.get("face_quality_face_sharpness_min", 0.75)),
            "face_quality_contrast_min": float(decision.get("face_quality_contrast_min", 0.55)),
            "face_quality_blur_min": float(decision.get("face_quality_blur_min", 0.55)),
            "face_quality_delta_face_sharpness_min": float(decision.get("face_quality_delta_face_sharpness_min", -10.0)),
            "face_quality_yaw_max_abs_deg": float(decision.get("face_quality_yaw_max_abs_deg", 30.0)),

            # expression の最低ライン
            "expression_min": float(decision.get("expression_min", 0.5)),

            # technical ルート
            "technical_contrast_min": float(decision.get("technical_contrast_min", 0.60)),
            "technical_blur_min": float(decision.get("technical_blur_min", 0.60)),
            "technical_delta_face_sharpness_min": float(decision.get("technical_delta_face_sharpness_min", -15.0)),
            "technical_exposure_min": float(decision.get("technical_exposure_min", 1.0)),
        }
        
    def evaluate(self) -> Dict[str, Any]:
        self.logger.info(f"評価開始: 画像ファイル {self.file_name}")
        results = {}

        try:
            # resized_2048 はRGBだけど、評価入力の必須はこっち
            if self.resized_2048_bgr_u8 is None:
                self.logger.error("resized_2048_bgr_u8 が None です。評価をスキップします。")
                return {}

            # 顔検出は RGB uint8 を使う（insightface 安定）
            face_result = {"faces": []}
            if self.skip_face_processing:
                self.logger.info("顔処理をスキップします。")
            else:
                face_result = self.face_processor.detect_faces(self.rgb_u8)

            faces = face_result.get("faces", [])
            results["face_detected"] = bool(faces)
            results["faces"] = faces

            # --- Full body / pose ---
            body_result: Dict[str, Any] = {}
            try:
                body_result = self.body_detector.detect(self.bgr_u8)
            except AttributeError:
                full_body = bool(self.body_detector.detect_full_body(self.bgr_u8))
                body_result = {
                    "full_body_detected": full_body,
                    "pose_score": 100.0 if full_body else 0.0,
                    "full_body_cut_risk": 0.0 if full_body else 1.0,
                }
            except Exception as e:
                self.logger.warning(f"全身検出に失敗: {e}")
                body_result = {
                    "full_body_detected": False,
                    "pose_score": 0.0,
                    "full_body_cut_risk": 1.0,
                }

            # ★必須キー穴埋め（detect() 実装の揺れを吸収）
            results.update(body_result)
            results.setdefault("full_body_detected", False)
            results.setdefault("pose_score", 0.0)
            results.setdefault("full_body_cut_risk", 1.0)

            # ★ body_keypoints を拾う（キー名は想定なので、存在しなければ []）
            body_keypoints = (
                body_result.get("keypoints")
                or body_result.get("body_keypoints")
                or []
            )

            # 構図評価
            composition_result = self._evaluate_composition(self.rgb_u8, faces, body_keypoints)
            results.update(composition_result)

            # 顔のスコア計算（顔全体に対して）
            results.update(self._evaluate_face(self.evaluators["face"], self.rgb_u8))

            best_face = self.face_processor.get_best_face(face_result["faces"])
            # 最も信頼性の高い顔に対する局所評価と属性追加
            if best_face:
                # crop はRGB u8から取る → 評価器用にBGRへ
                cropped_face_rgb_u8 = self.face_processor.crop_face(self.rgb_u8, best_face)
                cropped_face_bgr_u8 = cv2.cvtColor(cropped_face_rgb_u8, cv2.COLOR_RGB2BGR)

                face_attrs = self.face_processor.extract_attributes(best_face)
                results["yaw"] = face_attrs.get("yaw", 0)
                results["pitch"] = face_attrs.get("pitch", 0)
                results["roll"] = face_attrs.get("roll", 0)
                results["gaze"] = face_attrs.get("gaze", [0, 0])

                # ★ 追加: 顔ボックスの高さ比（0〜1）を計算して保存
                try:
                    h, _w = self.rgb_u8.shape[:2]
                    box = best_face.get("box")
                    if box and len(box) == 4 and h > 0:
                        x1, y1, x2, y2 = box
                        face_h = max(1.0, float(y2 - y1))
                        results["face_box_height_ratio"] = float(face_h / float(h))
                    else:
                        results["face_box_height_ratio"] = 0.0
                except Exception:
                    # 万が一 shape / box でコケても落ちないように
                    results["face_box_height_ratio"] = 0.0

                results["lead_room_score"] = self._calc_lead_room_score(
                    self.rgb_u8, best_face, results.get("yaw", 0.0)
                )

                results.update(self._evaluate_face_region(cropped_face_bgr_u8))
            else:
                # best_faceが無いときもキーは出しておくとCSVが安定
                results["yaw"] = 0
                results["pitch"] = 0
                results["roll"] = 0
                results["gaze"] = [0, 0]
                results["lead_room_score"] = 0.0
                # ★ 顔が取れないときは 0 扱い
                results["face_box_height_ratio"] = 0.0

            for key, evaluator in self.evaluators.items():
                if key == "face":
                    continue
                results.update(self._safe_evaluate(evaluator, self.resized_2048_bgr_u8, key))

            # ★ ここから追加: expression_score を組み立て
            self._add_expression_score(results)

            # 既存のデルタ系を追加
            self._add_face_global_deltas(results)

            accepted, reason = self.decide_accept_static(results, thresholds=self.decision_thresholds)
            results["accepted_flag"] = accepted
            results["accepted_reason"] = reason
            return results


        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}

    def _evaluate_face(self, evaluator, image):
        """
        顔検出および顔スコアの算出を行う。

        引数:
            evaluator: 顔評価用の Evaluator インスタンス
            image: 評価対象の画像（RGB形式）

        戻り値:
            dict: 顔スコアおよび検出顔情報
        """
        try:
            result = evaluator.evaluate(image)
            self.logger.info(f"顔評価結果: {result}")
            return result
        except Exception as e:
            self.logger.warning(f"顔評価中にエラー: {str(e)}")
            return {"face_score": 0, "faces": []}

    def _evaluate_composition(
        self,
        image: np.ndarray,
        face_boxes: list,
        body_keypoints: list,
    ) -> Dict[str, Any]:
        """
        顔と全身情報を基に構図の評価を実施する。

        引数:
            image (np.ndarray): 評価対象の画像（RGB形式）
            face_boxes (list): 顔のバウンディングボックスリスト
            body_keypoints (list): 全身のキーポイントリスト
        戻り値:
            Dict[str, Any]: 構図に関する各種スコアとグループID情報
                - composition_rule_based_score
                - face_position_score
                - framing_score
                - face_direction_score
                - eye_contact_score
                - face_composition_raw / face_composition_score
                - body_composition_raw / body_composition_score
                - composition_raw / composition_score / composition_status
                - group_id / subgroup_id
        """
        try:
            # 顔も全身も情報が全く無ければスキップ
            if not face_boxes and not body_keypoints:
                self.logger.warning("構図評価スキップ：face_boxes も body_keypoints も空です。")
                return {
                    "composition_rule_based_score": 0.0,
                    "face_position_score": 0.0,
                    "framing_score": 0.0,
                    "face_direction_score": 0.0,
                    "eye_contact_score": 0.0,
                    "face_composition_raw": None,
                    "face_composition_score": 0.5,  # ニュートラル
                    "body_composition_raw": None,
                    "body_composition_score": 0.5,
                    "composition_raw": None,
                    "composition_score": 0.5,
                    "composition_status": "not_computed_with_default",
                    "group_id": "unclassified",
                    "subgroup_id": -1,
                }

            # CompositeCompositionEvaluator に丸投げ
            result = self.composition_evaluator.evaluate(
                image=image,
                face_boxes=face_boxes,
                body_keypoints=body_keypoints,
            )
            self.logger.info(f"構図評価結果: {result}")

            return {
                # 顔ルールベースの旧最終スコア（連続値）※後方互換
                "composition_rule_based_score": result.get(
                    "composition_rule_based_score", 0.0
                ),
                "face_position_score": result.get("face_position_score", 0.0),
                "framing_score": result.get("framing_score", 0.0),
                "face_direction_score": result.get("face_direction_score", 0.0),
                "eye_contact_score": result.get("eye_contact_score", 0.0),

                # 新: 顔構図スコア
                "face_composition_raw": result.get("face_composition_raw"),
                "face_composition_score": result.get("face_composition_score"),

                # 新: 全身構図スコア
                "body_composition_raw": result.get("body_composition_raw"),
                "body_composition_score": result.get("body_composition_score"),

                # 新: 統合構図スコア
                "composition_raw": result.get("composition_raw"),
                "composition_score": result.get("composition_score"),
                "composition_status": result.get("composition_status", "ok"),

                # グループ分類（どちらかが上書きするが、それでOK）
                "group_id": result.get("group_id", "unclassified"),
                "subgroup_id": result.get("subgroup_id", -1),
            }

        except Exception as e:
            self.logger.warning(f"構図評価中にエラー: {str(e)}")
            return {
                "composition_rule_based_score": 0.0,
                "face_position_score": 0.0,
                "framing_score": 0.0,
                "face_direction_score": 0.0,
                "eye_contact_score": 0.0,
                "face_composition_raw": None,
                "face_composition_score": 0.5,
                "body_composition_raw": None,
                "body_composition_score": 0.5,
                "composition_raw": None,
                "composition_score": 0.5,
                "composition_status": "error_fallback",
                "group_id": "unclassified",
                "subgroup_id": -1,
            }
        

    def _safe_evaluate(self, evaluator, image: np.ndarray, name: str) -> Dict[str, Any]:
        """
        各評価器を安全に実行し、例外時にはデフォルト値を返す。

        引数:
            evaluator: 評価器インスタンス
            image (np.ndarray): 評価対象の画像
            name (str): 評価指標の名称（"sharpness" など）

        戻り値:
            Dict[str, Any]: 評価スコアおよび必要に応じた補助情報
        """
        try:
            result = evaluator.evaluate(image) or {}
            output: Dict[str, Any] = {}

            # --- NoiseEvaluator だけは raw 情報をそのまま通す ---
            if name == "noise":
                for k, v in result.items():
                    output[k] = v

            # --- SharpnessEvaluator（新仕様: raw + 離散スコア） ---
            elif name == "sharpness":
                # 0〜1 の離散スコア
                output["sharpness_score"] = result.get("sharpness_score", 0.5)

                # ラプラシアン分散の生値（デバッグ・後分析用）
                if "sharpness_raw" in result:
                    output["sharpness_raw"] = result["sharpness_raw"]

                # 評価ステータス（ok / invalid_input / error など）
                if "sharpness_eval_status" in result:
                    output["sharpness_eval_status"] = result["sharpness_eval_status"]

            # --- ContrastEvaluator（新仕様: raw + 離散スコア + grade） ---
            elif name == "contrast":
                # 0〜1 の離散スコア
                output["contrast_score"] = result.get("contrast_score", 0.5)

                # 生値（標準偏差）
                if "contrast_raw" in result:
                    output["contrast_raw"] = result["contrast_raw"]

                # grade (excellent/good/...）
                if "contrast_grade" in result:
                    output["contrast_grade"] = result["contrast_grade"]

            # --- BlurrinessEvaluator（新仕様: raw + 離散スコア + grade） ---
            elif name == "blurriness":
                # 0〜1 の離散スコア（意味スコア）
                output["blurriness_score"] = result.get("blurriness_score", 0.5)

                # 生値（統合ブレ指標）
                if "blurriness_raw" in result:
                    output["blurriness_raw"] = result["blurriness_raw"]

                # grade (excellent/good/fair/...)
                if "blurriness_grade" in result:
                    output["blurriness_grade"] = result["blurriness_grade"]

            # --- ExposureEvaluator（Noise と同ポリシー） ---
            elif name == "exposure":
                # 0〜1 の離散スコア
                output["exposure_score"] = result.get("exposure_score", 0.5)

                # grade があれば拾う
                if "exposure_grade" in result:
                    output["exposure_grade"] = result["exposure_grade"]

                # 生値（0..1 / 0..255 相当）
                if "mean_brightness" in result:
                    output["mean_brightness"] = result["mean_brightness"]
                if "mean_brightness_8bit" in result:
                    output["mean_brightness_8bit"] = result["mean_brightness_8bit"]

                # 評価ステータス系
                if "exposure_eval_status" in result:
                    output["exposure_eval_status"] = result["exposure_eval_status"]
                if "exposure_fallback_reason" in result:
                    output["exposure_fallback_reason"] = result["exposure_fallback_reason"]

                # 必要なら dtype も（デバッグ用）
                if "image_dtype" in result:
                    output["exposure_image_dtype"] = result["image_dtype"]

            else:
                # 従来どおりの 1スコア系 evaluator
                output[f"{name}_score"] = result.get(f"{name}_score", 0)

                # local_* 系は std も出す
                if "local_" in name:
                    output[f"{name}_std"] = result.get(f"{name}_std", 0)

            gc.collect()
            return output

        except Exception as e:
            self.logger.error(f"{name} 評価中にエラー: {str(e)}")
            self.logger.error(traceback.format_exc())

            # Noise のときだけ、フォールバック値を少し丁寧に返す
            if name == "noise":
                return {
                    "noise_score": 0.5,                 # 中立寄り
                    "noise_grade": "error",
                    "noise_sigma_midtone": None,
                    "noise_sigma_used": None,
                    "noise_mask_ratio": None,
                    "noise_eval_status": "fallback",
                    "noise_fallback_reason": "exception",
                }

            # Sharpness のフォールバックも少し丁寧に返す
            if name == "sharpness":
                return {
                    "sharpness_score": 0.5,              # 中立寄り
                    "sharpness_raw": None,
                    "sharpness_eval_status": "exception",
                }

            # ★ Exposure のフォールバックも丁寧に返す
            if name == "exposure":
                return {
                    "exposure_score": 0.5,
                    "exposure_grade": "error",
                    "mean_brightness": None,
                    "mean_brightness_8bit": None,
                    "exposure_eval_status": "fallback",
                    "exposure_fallback_reason": "exception",
                }

            # それ以外は従来どおり
            return {f"{name}_score": 0}


    def _evaluate_face_region(self, face_crop: np.ndarray) -> Dict[str, Any]:
        """
        顔領域に対して個別評価器を適用し、局所スコアを算出する。
        """
        face_scores: Dict[str, Any] = {}

        for key in [
            "sharpness",
            "contrast",
            "noise",
            "blurriness",
            "local_sharpness",
            "local_contrast",
            "exposure",
            "color_balance",
        ]:
            evaluator = self.evaluators.get(key)
            if evaluator is None:
                continue

            try:
                result = evaluator.evaluate(face_crop) or {}

                # ---------- ノイズだけは新仕様にあわせて特別扱い ----------
                if key == "noise":
                    # 5段階(0〜1)のスコアをそのまま顔用に流用
                    face_scores["face_noise_score"] = result.get("noise_score", 0.5)

                    # 必要なら grade や生値も顔用に残しておく
                    if "noise_grade" in result:
                        face_scores["face_noise_grade"] = result["noise_grade"]
                    if "noise_sigma_midtone" in result:
                        face_scores["face_noise_sigma_midtone"] = result["noise_sigma_midtone"]
                    if "noise_sigma_used" in result:
                        face_scores["face_noise_sigma_used"] = result["noise_sigma_used"]
                    if "noise_mask_ratio" in result:
                        face_scores["face_noise_mask_ratio"] = result["noise_mask_ratio"]
                    if "noise_eval_status" in result:
                        face_scores["face_noise_eval_status"] = result["noise_eval_status"]
                    if "noise_fallback_reason" in result:
                        face_scores["face_noise_fallback_reason"] = result["noise_fallback_reason"]

                    # ノイズはこれで次の key へ
                    continue

                # ---------- SharpnessEvaluator（raw + 離散スコア） ----------
                if key == "sharpness":
                    # 意味スコア（0〜1 の 5段階）
                    face_scores["face_sharpness_score"] = result.get("sharpness_score", 0.5)

                    # 生値（ラプラシアン分散）はあれば拾うだけ（ヘッダが無ければ CSV 側で捨てられる）
                    if "sharpness_raw" in result:
                        face_scores["face_sharpness_raw"] = result["sharpness_raw"]

                    # 評価ステータスもあれば保存
                    if "sharpness_eval_status" in result:
                        face_scores["face_sharpness_eval_status"] = result["sharpness_eval_status"]

                    # sharpness はここで完了
                    continue

                # ---------- ContrastEvaluator（顔用） ----------
                if key == "contrast":
                    face_scores["face_contrast_score"] = result.get("contrast_score", 0.5)

                    if "contrast_raw" in result:
                        face_scores["face_contrast_raw"] = result["contrast_raw"]
                    if "contrast_grade" in result:
                        face_scores["face_contrast_grade"] = result["contrast_grade"]

                    continue

                # ---------- BlurrinessEvaluator（顔用） ----------
                if key == "blurriness":
                    face_scores["face_blurriness_score"] = result.get("blurriness_score", 0.5)

                    if "blurriness_raw" in result:
                        face_scores["face_blurriness_raw"] = result["blurriness_raw"]
                    if "blurriness_grade" in result:
                        face_scores["face_blurriness_grade"] = result["blurriness_grade"]

                    continue

                # ---------- ExposureEvaluator（顔用） ----------
                if key == "exposure":
                    # 意味スコア
                    face_scores["face_exposure_score"] = result.get("exposure_score", 0.5)

                    # grade
                    if "exposure_grade" in result:
                        face_scores["face_exposure_grade"] = result["exposure_grade"]

                    # 生値
                    if "mean_brightness" in result:
                        face_scores["face_mean_brightness"] = result["mean_brightness"]
                    if "mean_brightness_8bit" in result:
                        face_scores["face_mean_brightness_8bit"] = result["mean_brightness_8bit"]

                    # ステータス系
                    if "exposure_eval_status" in result:
                        face_scores["face_exposure_eval_status"] = result["exposure_eval_status"]
                    if "exposure_fallback_reason" in result:
                        face_scores["face_exposure_fallback_reason"] = result["exposure_fallback_reason"]

                    continue

                # ---------- それ以外（従来どおり） ----------
                score_key = f"face_{key}_score"
                face_scores[score_key] = result.get(f"{key}_score", 0)

                # local_* 系は std も拾う
                if "local_" in key:
                    face_scores[f"face_{key}_std"] = result.get(f"{key}_std", 0)

            except Exception as e:
                self.logger.warning(f"顔領域の{key}評価に失敗: {str(e)}")

                if key == "noise":
                    # ノイズ評価の例外時はニュートラル寄りで返す
                    face_scores["face_noise_score"] = 0.5
                    face_scores["face_noise_grade"] = "warn"

                elif key == "sharpness":
                    # シャープネス評価の例外時もニュートラル寄りにしておく
                    face_scores["face_sharpness_score"] = 0.5
                    face_scores["face_sharpness_eval_status"] = "exception"

                elif key == "blurriness":
                    face_scores["face_blurriness_score"] = 0.5
                    face_scores["face_blurriness_grade"] = "warn"

                # ★ Exposure の例外時はニュートラル寄りで返す
                elif key == "exposure":
                    face_scores["face_exposure_score"] = 0.5
                    face_scores["face_exposure_grade"] = "warn"

                else:
                    face_scores[f"face_{key}_score"] = 0

        return face_scores


    # =========================
    # utility
    # =========================

    def _calc_lead_room_score(self, image: np.ndarray, best_face: Dict[str, Any], yaw: float) -> float:
        """
        視線（yaw）の向きに余白があるほど +、逆なら -。
        返り値: [-1, 1]

        best_face["box"] は [x1, y1, x2, y2] を想定。
        """
        h, W = image.shape[:2]
        box = best_face.get("box")
        if not box or len(box) != 4:
            return 0.0

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0

        left_space = cx
        right_space = W - cx

        if yaw > 5:       # 右向き想定
            raw = (right_space - left_space) / W
        elif yaw < -5:    # 左向き想定
            raw = (left_space - right_space) / W
        else:
            raw = 0.0

        return float(max(-1.0, min(1.0, raw)))


    def _add_face_global_deltas(self, results: Dict[str, Any]) -> None:
        def d(a, b):
            if a is None or b is None:
                return None
            try:
                return float(a) - float(b)
            except (TypeError, ValueError):
                return None

        results["delta_face_sharpness"] = d(
            results.get("face_sharpness_score"), results.get("sharpness_score")
        )
        results["delta_face_contrast"] = d(
            results.get("face_contrast_score"), results.get("contrast_score")
        )


    def _add_expression_score(self, results: Dict[str, Any]) -> None:
        """
        既存の face 向き系の指標から簡易 expression_score (0〜1) を作る。

        - eye_contact_score / face_direction_score をベースに加点
        - yaw / pitch が大きいと減点
        """

        def _f(key: str, default: float = 0.0) -> float:
            v = results.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        eye_contact = _f("eye_contact_score", 0.0)
        face_direction = _f("face_direction_score", 0.0)
        yaw = abs(_f("yaw", 0.0))
        pitch = abs(_f("pitch", 0.0))

        # ベースは「目線＋顔の向き」
        base = 0.6 * eye_contact + 0.4 * face_direction  # 0〜1 前提

        # yaw/pitch が大きすぎると減点（45度超えたら最大ペナルティ）
        yaw_penalty = min(yaw / 45.0, 1.0)
        pitch_penalty = min(pitch / 45.0, 1.0)

        # ペナルティを 0〜1 にまとめる（0: 正面〜少し横, 1: 真横＆うつむき/のけ反り）
        penalty = 0.5 * yaw_penalty + 0.5 * pitch_penalty

        # ペナルティを 0〜0.4 くらいの係数で効かせる（お好みで調整可）
        score = base * (1.0 - 0.4 * penalty)

        # クリップ
        score = max(0.0, min(1.0, score))

        # グレード分類
        if score >= 0.9:
            grade = "excellent"
        elif score >= 0.75:
            grade = "good"
        elif score >= 0.5:
            grade = "fair"
        elif score >= 0.25:
            grade = "poor"
        else:
            grade = "bad"

        results["expression_score"] = score
        results["expression_grade"] = grade

    @staticmethod
    def decide_accept_static(
        results: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        return decide_accept(results, thresholds=thresholds)

