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
            "face": FaceEvaluator(backend="insightface"),
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

            # full body
            "full_body_pose_min_100": float(decision.get("full_body_pose_min_100", 55.0)),
            "full_body_cut_risk_max": float(decision.get("full_body_cut_risk_max", 0.6)),
            "blurriness_min_full_body": float(decision.get("blurriness_min_full_body", 0.45)),
            "exposure_min_common": float(decision.get("exposure_min_common", 0.5)),

            # composition ルート
            "composition_score_min": float(decision.get("composition_score_min", 0.75)),
            "framing_score_min": float(decision.get("framing_score_min", 0.5)),
            "lead_room_score_min": float(decision.get("lead_room_score_min", 0.10)),

            # face_quality ルート
            "face_quality_exposure_min": float(decision.get("face_quality_exposure_min", 0.5)),
            "face_quality_face_sharpness_min": float(decision.get("face_quality_face_sharpness_min", 0.75)),
            "face_quality_contrast_min": float(decision.get("face_quality_contrast_min", 55.0)),
            "face_quality_blur_min": float(decision.get("face_quality_blur_min", 0.55)),
            "face_quality_delta_face_sharpness_min": float(decision.get("face_quality_delta_face_sharpness_min", -10.0)),
            "face_quality_yaw_max_abs_deg": float(decision.get("face_quality_yaw_max_abs_deg", 30.0)),

            # technical ルート
            "technical_contrast_min": float(decision.get("technical_contrast_min", 60.0)),
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

            # 構図評価(１回目)
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

            composition_result = self._evaluate_composition(
                self.rgb_image, face_result.get("faces", []), body_keypoints
            )
            results.update(composition_result)

            for key, evaluator in self.evaluators.items():
                if key == "face":
                    continue
                results.update(self._safe_evaluate(evaluator, self.resized_2048_bgr_u8, key))

            self._add_face_global_deltas(results)
            accepted, reason = self._decide_accept(results)
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
                # NoiseEvaluator 側で最終キー名を決めている前提なので、
                # ここでは一切リネームせずにそのまま流す。
                # 例:
                #   noise_score
                #   noise_grade
                #   noise_sigma_midtone
                #   noise_sigma_used
                #   noise_mask_ratio
                #   noise_eval_status
                #   noise_fallback_reason
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

            else:
                # 従来どおりの 1スコア系 evaluator
                output[f"{name}_score"] = result.get(f"{name}_score", 0)

                # local_* 系は std も出す
                if "local_" in name:
                    output[f"{name}_std"] = result.get(f"{name}_std", 0)

                # 露出だけ mean_brightness を追加
                if name == "exposure":
                    output["mean_brightness"] = result.get("mean_brightness", 0)

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

                # ---------- それ以外（従来どおり） ----------
                score_key = f"face_{key}_score"
                face_scores[score_key] = result.get(f"{key}_score", 0)

                # local_* 系は std も拾う
                if "local_" in key:
                    face_scores[f"face_{key}_std"] = result.get(f"{key}_std", 0)

                # 露出のみ mean_brightness も拾う
                if key == "exposure":
                    face_scores["face_mean_brightness"] = result.get("mean_brightness", 0)

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
        """
        h, W = image.shape[:2]
        box = best_face.get("box")  # [x, y, w, h]
        if not box or len(box) != 4:
            return 0.0

        x, y, w, h = box
        cx = x + w / 2.0
        left_space = cx
        right_space = W - cx

        # yaw の符号は環境で逆のことがあるので、まず閾値だけ置く（微小は0）
        if yaw > 5:        # 右向き想定
            raw = (right_space - left_space) / W
        elif yaw < -5:     # 左向き想定
            raw = (left_space - right_space) / W
        else:
            raw = 0.0

        # 安全にクリップ
        if raw > 1.0:
            raw = 1.0
        elif raw < -1.0:
            raw = -1.0
        return float(raw)


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


    @staticmethod
    def decide_accept_static(
        results: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None,  # ★ 追加
    ) -> Tuple[bool, str]:
        """
        ノイズスコアが 0〜1 の 5段階離散 (1.0, 0.75, 0.5, 0.25, 0.0) を前提とした判定。

        旧ロジックの 60 / 70 しきい値は以下のようにマッピング:
            60  → 0.5  (fair 以上)
            70  → 0.75 (good 以上)

        さらに noise_grade / face_noise_grade も補助的に参照する:
            excellent → 1.0
            good      → 0.75
            fair/warn → 0.5
            poor      → 0.25
            bad       → 0.0
        """

        def _f(key: str, default: float = 0.0) -> float:
            v = results.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        def _pose_to_100(v: float) -> float:
            """
            pose_score が 0〜1 / 0〜100 どちらでも来てもよいように正規化する。
            """
            try:
                v = float(v)
            except (TypeError, ValueError):
                return 0.0
            if 0.0 <= v <= 1.0:
                return v * 100.0
            return v

        def _grade_to_score_like(grade: str) -> float | None:
            """
            grade 文字列から 0〜1 のスコア相当値にマッピング。
            未知のラベルは None（無視）とする。
            """
            if not grade:
                return None
            g = grade.strip().lower()

            if g in ("excellent", "very_good", "best"):
                return 1.0
            if g in ("good",):
                return 0.75
            if g in ("fair", "ok", "warn"):
                return 0.5
            if g in ("poor", "weak"):
                return 0.25
            if g in ("bad", "ng"):
                return 0.0
            return None

        def _ok_from(score: float, grade_score: float | None, thr: float) -> bool:
            """
            数値スコア or グレード由来スコアのどちらかが閾値を超えれば OK。
            """
            if score >= thr:
                return True
            if grade_score is not None and grade_score >= thr:
                return True
            return False

        def _thr(name: str, default: float) -> float:
            """
            閾値テーブルから取り出し。なければデフォルトを使う。
            """
            if thresholds is None:
                return default
            try:
                return float(thresholds.get(name, default))
            except (TypeError, ValueError):
                return default

        # 共通で使う数値を先に float 化
        noise_score = _f("noise_score", 0.0)
        face_noise_score = _f("face_noise_score", 0.0)
        blurriness_score = _f("blurriness_score", 0.0)
        exposure_score = _f("exposure_score", 0.0)
        pose_score_raw = _f("pose_score", 0.0)
        pose_score_100 = _pose_to_100(pose_score_raw)
        full_body_cut_risk = _f("full_body_cut_risk", 1.0)
        yaw = abs(_f("yaw", 0.0))

        contrast_score = _f("contrast_score", 0.0)

        # ★ 新スケール: まず composition_score (0〜1) を使い、
        #   無ければ従来の composition_rule_based_score をフォールバックで使う
        raw_comp_score = results.get("composition_score")
        if raw_comp_score is not None:
            try:
                composition_score = float(raw_comp_score)
            except (TypeError, ValueError):
                composition_score = _f("composition_rule_based_score", 0.0)
        else:
            composition_score = _f("composition_rule_based_score", 0.0)

        framing_score = _f("framing_score", 0.0)
        lead_room_score = _f("lead_room_score", 0.0)

        delta_face_sharpness = _f("delta_face_sharpness", -999.0)
        face_sharpness_score = _f("face_sharpness_score", 0.0)

        full_body_detected = bool(results.get("full_body_detected"))
        face_detected = bool(results.get("face_detected"))

        # grade 情報も読む
        noise_grade = str(results.get("noise_grade", "") or "")
        face_noise_grade = str(results.get("face_noise_grade", "") or "")

        noise_grade_score = _grade_to_score_like(noise_grade)
        face_noise_grade_score = _grade_to_score_like(face_noise_grade)

        # --- ノイズ関連の閾値 ---
        noise_ok_thr = _thr("noise_ok", 0.5)
        noise_good_thr = _thr("noise_good", 0.75)
        face_noise_good_thr = _thr("face_noise_good", 0.75)

        noise_ok = _ok_from(noise_score, noise_grade_score, noise_ok_thr)
        noise_good = _ok_from(noise_score, noise_grade_score, noise_good_thr)
        face_noise_good = _ok_from(face_noise_score, face_noise_grade_score, face_noise_good_thr)

        # full body 共通条件（pose_score は 0〜100 換算で扱う）
        pose_min_100 = _thr("full_body_pose_min_100", 55.0)
        cut_risk_max = _thr("full_body_cut_risk_max", 0.6)
        blurriness_min_full = _thr("blurriness_min_full_body", 0.45)
        exposure_min_common = _thr("exposure_min_common", 0.5)

        full_body_ok = (
            full_body_detected
            and pose_score_100 >= pose_min_100
            and full_body_cut_risk <= cut_risk_max
            and noise_ok
            and blurriness_score >= blurriness_min_full
            and exposure_score >= exposure_min_common
        )

        # --- Full body route: 顔が無くても全身が検出できていればルート判定へ ---
        if not face_detected:
            if full_body_detected:
                cut_risk = full_body_cut_risk

                # 全身ルート（顔なしでもOK）
                if (
                    pose_score_100 >= pose_min_100
                    and cut_risk <= cut_risk_max
                    and noise_ok
                    and blurriness_score >= blurriness_min_full
                    and exposure_score >= exposure_min_common
                ):
                    return True, "full_body"

                return False, "full_body_rejected"

            return False, "no_face"

        # ==== ここから顔ありルート ====
        # face_quality route 用の閾値
        fq_exposure_min = _thr("face_quality_exposure_min", 0.5)
        fq_face_sharpness_min = _thr("face_quality_face_sharpness_min", 0.75)  # ← 0.75 に修正
        fq_contrast_min = _thr("face_quality_contrast_min", 55.0)
        fq_blur_min = _thr("face_quality_blur_min", 0.55)
        fq_delta_sharpness_min = _thr("face_quality_delta_face_sharpness_min", -10.0)
        fq_yaw_max = _thr("face_quality_yaw_max_abs_deg", 30.0)

        # Route C: face quality（最優先）
        if (
            exposure_score >= fq_exposure_min
            and face_sharpness_score >= fq_face_sharpness_min
            and face_noise_good
            and contrast_score >= fq_contrast_min
            and blurriness_score >= fq_blur_min
            and delta_face_sharpness >= fq_delta_sharpness_min
            and yaw <= fq_yaw_max
        ):
            return True, "face_quality"

        # Route D: full body（顔が弱くても拾う）
        if full_body_ok:
            return True, "full_body"

        # Route A: composition
        composition_score_min = _thr("composition_score_min", 0.75)
        framing_score_min = _thr("framing_score_min", 0.5)
        lead_room_min = _thr("lead_room_score_min", 0.10)

        if (
            composition_score >= composition_score_min
            and framing_score >= framing_score_min
            and lead_room_score >= lead_room_min
            and noise_ok
            and blurriness_score >= blurriness_min_full
            and exposure_score >= exposure_min_common
        ):
            return True, "composition"

        tech_contrast_min = _thr("technical_contrast_min", 60.0)
        tech_blur_min = _thr("technical_blur_min", 0.60)
        tech_delta_sharpness_min = _thr("technical_delta_face_sharpness_min", -15.0)
        tech_exposure_min = _thr("technical_exposure_min", 1.0)

        # Route B: technical
        if (
            noise_good
            and contrast_score >= tech_contrast_min
            and blurriness_score >= tech_blur_min
            and delta_face_sharpness >= tech_delta_sharpness_min
            and exposure_score >= tech_exposure_min
        ):
            return True, "technical"

        return False, "rejected"

    def _decide_accept(self, results: Dict[str, Any]) -> Tuple[bool, str]:
        # YAML からロードした閾値テーブルを static メソッドに渡す
        return self.decide_accept_static(results, thresholds=self.decision_thresholds)

