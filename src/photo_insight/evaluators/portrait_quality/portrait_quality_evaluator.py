# src/evaluators/portrait_quality/portrait_quality_evaluator.py
# -*- coding: utf-8 -*-

import cv2
import gc
import numpy as np
import os
import traceback
import yaml

from typing import Optional, Tuple, Dict, Any

# 自作ライブラリ（変更不可）
from photo_insight.evaluators.face_evaluator import FaceEvaluator
from photo_insight.evaluators.sharpness_evaluator import SharpnessEvaluator
from photo_insight.evaluators.blurriness_evaluator import BlurrinessEvaluator
from photo_insight.evaluators.contrast_evaluator import ContrastEvaluator
from photo_insight.evaluators.composite_composition_evaluator import CompositeCompositionEvaluator
from photo_insight.evaluators.noise_evaluator import NoiseEvaluator
from photo_insight.evaluators.exposure_evaluator import ExposureEvaluator
from photo_insight.evaluators.local_sharpness_evaluator import LocalSharpnessEvaluator
from photo_insight.evaluators.local_contrast_evaluator import LocalContrastEvaluator
from photo_insight.evaluators.rule_based_composition_evaluator import RuleBasedCompositionEvaluator
from photo_insight.evaluators.color_balance_evaluator import ColorBalanceEvaluator
from photo_insight.detectors.body_detection import FullBodyDetector
from photo_insight.utils.app_logger import Logger

from photo_insight.image_utils.image_preprocessor import ImagePreprocessor
from photo_insight.face_detectors.face_processor import FaceProcessor

from photo_insight.evaluators.quality_thresholds import QualityThresholds
from photo_insight.evaluators.portrait_accept_rules import decide_accept
from photo_insight.evaluators.portrait_quality.metric_mapping import MetricResultMapper

from photo_insight.evaluators.common.grade_contract import (
    STATUS_NOT_COMPUTED,
    STATUS_FALLBACK,
    STATUS_OK,
    normalize_eval_status,
)


GLOBAL_METRICS = (
    "sharpness",
    "blurriness",
    "contrast",
    "noise",
    "local_sharpness",
    "local_contrast",
    "exposure",
    "color_balance",
)

FACE_REGION_METRICS = (
    "sharpness",
    "blurriness",
    "contrast",
    "noise",
    "local_sharpness",
    "local_contrast",
    "exposure",
    "color_balance",
)


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

        # --- evaluator 用の閾値/設定（discretize_thresholds_raw 等）をロード ---
        self.eval_config = self._load_evaluator_config()

        # -------------------------
        # helpers: face_* -> base metric config injection
        # -------------------------
        def _subcfg(metric_key: str) -> Dict[str, Any]:
            """
            metric_key: "face_blurriness" 等
            戻り値: evaluator が期待する base key に寄せた dict
              例: {"blurriness": cfg["face_blurriness"]}
            """
            cfg = self.eval_config if isinstance(self.eval_config, dict) else {}
            spec = cfg.get(metric_key)
            if not isinstance(spec, dict):
                return {}
            base = metric_key.replace("face_", "")
            return {base: spec}

        # 前処理器で画像を一括取得
        self.face_evaluator = FaceEvaluator(backend="insightface")

        # ★ 重要: 引数で face_processor が渡されていればそれを優先（テストDI/差し替え用）
        self.face_processor = face_processor or FaceProcessor(self.face_evaluator, logger=self.logger)

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

        # -------------------------
        # evaluators: global / face (separated configs)
        # -------------------------
        # global: use full config as-is
        self.evaluators_global = {
            "face": self.face_evaluator,
            "sharpness": SharpnessEvaluator(logger=self.logger, config=self.eval_config),
            "blurriness": BlurrinessEvaluator(logger=self.logger, config=self.eval_config),
            "contrast": ContrastEvaluator(logger=self.logger, config=self.eval_config, metric_key="contrast"),
            "noise": NoiseEvaluator(
                max_noise_value=max_noise_value,  # 互換で残してOK
                logger=self.logger,
                config=self.eval_config,
            ),
            "local_sharpness": LocalSharpnessEvaluator(logger=self.logger, config=self.eval_config),
            "local_contrast": LocalContrastEvaluator(),
            "exposure": ExposureEvaluator(),
            "color_balance": ColorBalanceEvaluator(),
        }

        # face: inject face_* thresholds into base metric key expected by evaluator
        # 例: face_blurriness -> {"blurriness": {...}}
        self.evaluators_face = {
            "face": self.face_evaluator,
            "sharpness": SharpnessEvaluator(logger=self.logger, config=_subcfg("face_sharpness")),
            "blurriness": BlurrinessEvaluator(logger=self.logger, config=_subcfg("face_blurriness")),
            "contrast": ContrastEvaluator(logger=self.logger, config=_subcfg("face_contrast"), metric_key="contrast"),
            "noise": NoiseEvaluator(
                max_noise_value=max_noise_value,
                logger=self.logger,
                config=_subcfg("face_noise"),
            ),
            "local_sharpness": LocalSharpnessEvaluator(logger=self.logger, config=_subcfg("face_local_sharpness")),
            "local_contrast": LocalContrastEvaluator(),
            "exposure": ExposureEvaluator(),
            "color_balance": ColorBalanceEvaluator(),
        }

        self.body_detector = FullBodyDetector()
        self.mapper = MetricResultMapper()

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
        results: Dict[str, Any] = {}

        try:
            if self.resized_2048_bgr_u8 is None:
                self.logger.error("resized_2048_bgr_u8 が None です。評価をスキップします。")
                return {}

            # -------------------------
            # face detect
            # -------------------------
            face_result = {"faces": []}
            if self.skip_face_processing:
                self.logger.info("顔処理をスキップします。")
            else:
                face_result = self.face_processor.detect_faces(self.rgb_u8) or {"faces": []}

            faces = face_result.get("faces", []) or []
            results["face_detected"] = bool(faces)
            results["faces"] = faces

            # -------------------------
            # full body / pose
            # -------------------------
            try:
                body_result = self.body_detector.detect(self.bgr_u8) or {}
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

            results.update(body_result)

            body_keypoints = body_result.get("keypoints") or body_result.get("body_keypoints") or []

            # -------------------------
            # composition
            # -------------------------
            results.update(self._evaluate_composition(self.rgb_u8, faces, body_keypoints))

            # -------------------------
            # face attributes + face region metrics
            # -------------------------
            best_face = self.face_processor.get_best_face(faces)
            if best_face:
                cropped_face_rgb_u8 = self.face_processor.crop_face(self.rgb_u8, best_face)
                if cropped_face_rgb_u8 is not None:
                    cropped_face_bgr_u8 = cv2.cvtColor(cropped_face_rgb_u8, cv2.COLOR_RGB2BGR)

                    face_attrs = self.face_processor.extract_attributes(best_face) or {}
                    results["yaw"] = face_attrs.get("yaw", 0.0)
                    results["pitch"] = face_attrs.get("pitch", 0.0)
                    results["roll"] = face_attrs.get("roll", 0.0)
                    results["gaze"] = face_attrs.get("gaze", [0, 0])

                    # face_box_height_ratio
                    try:
                        h, _w = self.rgb_u8.shape[:2]
                        box = best_face.get("box") or best_face.get("bbox")
                        if box and len(box) == 4 and h > 0:
                            x1, y1, x2, y2 = box
                            face_h = max(1.0, float(y2 - y1))
                            results["face_box_height_ratio"] = float(face_h / float(h))
                        else:
                            results["face_box_height_ratio"] = 0.0
                    except Exception:
                        results["face_box_height_ratio"] = 0.0

                    results["lead_room_score"] = self._calc_lead_room_score(
                        self.rgb_u8, best_face, float(results.get("yaw", 0.0))
                    )

                    results.update(
                        self._eval_metrics(
                            cropped_face_bgr_u8,
                            FACE_REGION_METRICS,
                            prefix="face_",
                            tag="face",
                            evaluators_dict=self.evaluators_face,
                        )
                    )
                else:
                    # 顔は検出されたが crop に失敗したケース：ログだけ残して global へ
                    self.logger.warning("Face detected but crop_face returned None. Skip face-region metrics.")

            # -------------------------
            # global metrics
            # -------------------------
            results.update(
                self._eval_metrics(
                    self.resized_2048_bgr_u8,
                    GLOBAL_METRICS,
                    prefix="",
                    tag="global",
                    evaluators_dict=self.evaluators_global,
                )
            )

            # -------------------------
            # derived
            # -------------------------
            self._add_expression_score(results)
            self._add_face_global_deltas(results)

            accepted, reason = self.decide_accept_static(results, thresholds=self.decision_thresholds)
            results["accepted_flag"] = accepted
            results["accepted_reason"] = reason

            # -------------------------
            # finalize (schema + csv normalization)
            # -------------------------
            self._ensure_result_schema(results)
            self._normalize_for_csv(results)

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
        """
        def _empty_payload(status: str) -> Dict[str, Any]:
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
                # ★ SSOT
                "composition_status": status,
                "main_subject_center_source": None,
                "main_subject_center_x": None,
                "main_subject_center_y": None,
                "rule_of_thirds_raw": None,
                "rule_of_thirds_score": None,
                "contrib_comp_composition_rule_based_score": None,
                "contrib_comp_face_position_score": None,
                "contrib_comp_framing_score": None,
                "contrib_comp_lead_room_score": None,
                "contrib_comp_body_composition_score": None,
                "contrib_comp_rule_of_thirds_score": None,
                "group_id": "unclassified",
                "subgroup_id": -1,
            }

        try:
            # 顔も全身も情報が全く無ければスキップ
            if not face_boxes and not body_keypoints:
                self.logger.warning("構図評価スキップ：face_boxes も body_keypoints も空です。")
                return _empty_payload(status=STATUS_NOT_COMPUTED)

            # CompositeCompositionEvaluator に丸投げ
            result = self.composition_evaluator.evaluate(
                image=image,
                face_boxes=face_boxes,
                body_keypoints=body_keypoints,
            ) or {}
            self.logger.info(f"構図評価結果: {result}")

            # ここで出てくる status も SSOT に寄せておく（未知値の揺れ防止）
            status = normalize_eval_status(result.get("composition_status", STATUS_OK))

            return {
                "composition_rule_based_score": result.get("composition_rule_based_score", 0.0),
                "face_position_score": result.get("face_position_score", 0.0),
                "framing_score": result.get("framing_score", 0.0),
                "face_direction_score": result.get("face_direction_score", 0.0),
                "eye_contact_score": result.get("eye_contact_score", 0.0),

                "face_composition_raw": result.get("face_composition_raw"),
                "face_composition_score": result.get("face_composition_score"),

                "body_composition_raw": result.get("body_composition_raw"),
                "body_composition_score": result.get("body_composition_score"),

                "composition_raw": result.get("composition_raw"),
                "composition_score": result.get("composition_score"),
                "composition_status": status,

                "main_subject_center_source": result.get("main_subject_center_source"),
                "main_subject_center_x": result.get("main_subject_center_x"),
                "main_subject_center_y": result.get("main_subject_center_y"),

                "rule_of_thirds_raw": result.get("rule_of_thirds_raw"),
                "rule_of_thirds_score": result.get("rule_of_thirds_score"),

                "contrib_comp_composition_rule_based_score": result.get("contrib_comp_composition_rule_based_score"),
                "contrib_comp_face_position_score": result.get("contrib_comp_face_position_score"),
                "contrib_comp_framing_score": result.get("contrib_comp_framing_score"),
                "contrib_comp_lead_room_score": result.get("contrib_comp_lead_room_score"),
                "contrib_comp_body_composition_score": result.get("contrib_comp_body_composition_score"),
                "contrib_comp_rule_of_thirds_score": result.get("contrib_comp_rule_of_thirds_score"),

                "group_id": result.get("group_id", "unclassified"),
                "subgroup_id": result.get("subgroup_id", -1),
            }

        except Exception as e:
            self.logger.warning(f"構図評価中にエラー: {str(e)}")
            return _empty_payload(status=STATUS_FALLBACK)

    def _evaluate_face_region(self, face_crop_bgr_u8: np.ndarray) -> Dict[str, Any]:
        """
        顔領域に対して個別評価器を適用し、局所スコアを算出する。
        """
        out: Dict[str, Any] = {}
        for name in FACE_REGION_METRICS:
            evaluator = self.evaluators_face.get(name)
            if evaluator is None:
                continue
            r = self._try_eval(evaluator, face_crop_bgr_u8, name=f"face:{name}")
            out.update(self.mapper.map(name, r, prefix="face_"))
        return out

    # =========================
    # utility
    # =========================
    def _eval_metrics(self, image: np.ndarray, metrics, prefix: str, tag: str, evaluators_dict) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in metrics:
            evaluator = evaluators_dict.get(name)
            if evaluator is None:
                continue
            r = self._try_eval(evaluator, image, name=f"{tag}:{name}")
            out.update(self.mapper.map(name, r, prefix=prefix))
        return out

    def _ensure_result_schema(self, results: Dict[str, Any]) -> None:
        """
        CSVスキーマを安定化させるための穴埋め。
        - 顔が無い場合: face_* 系は not_computed_with_default / no_face に統一
        - 顔がある場合: 既存値を尊重しつつ欠損だけ埋める
        """
        results.setdefault("face_detected", False)
        results.setdefault("faces", [])
        results.setdefault("yaw", 0.0)
        results.setdefault("pitch", 0.0)
        results.setdefault("roll", 0.0)
        results.setdefault("gaze", [0, 0])
        results.setdefault("lead_room_score", 0.0)
        results.setdefault("face_box_height_ratio", 0.0)

        results.setdefault("full_body_detected", False)
        results.setdefault("pose_score", 0.0)
        results.setdefault("full_body_cut_risk", 1.0)

        if "body_center_y_ratio" in results and "body_center_y" not in results:
            results["body_center_y"] = results.get("body_center_y_ratio")

        results.setdefault("accepted_flag", False)
        results["accepted_reason"] = str(results.get("accepted_reason") or "")
        results.setdefault("shot_type", "unknown")

        face_detected = bool(results.get("face_detected"))

        face_metrics = (
            "sharpness",
            "blurriness",
            "contrast",
            "noise",
            "local_sharpness",
            "local_contrast",
            "exposure",
            "color_balance",
        )

        default_score = {
            "sharpness": 0.5,
            "blurriness": 0.5,
            "contrast": 0.5,
            "noise": 0.5,
            "local_sharpness": 0.0,
            "local_contrast": 0.0,
            "exposure": 0.5,
            "color_balance": 0.5,
        }

        if not face_detected:
            status = STATUS_NOT_COMPUTED
            reason = "no_face"

            for m in face_metrics:
                results.setdefault(f"face_{m}_score", default_score.get(m, 0.0))
                results[f"face_{m}_eval_status"] = status
                results.setdefault(f"face_{m}_fallback_reason", reason)

            results.setdefault("face_blurriness_grade", None)
            results.setdefault("face_contrast_grade", None)
            results.setdefault("face_noise_grade", None)
            results.setdefault("face_exposure_grade", None)
            results.setdefault("face_color_balance_grade", None)

            results.setdefault("face_local_sharpness_std", 0.0)
            results.setdefault("face_local_contrast_std", 0.0)

            results.setdefault("face_sharpness_raw", None)
            results.setdefault("face_blurriness_raw", None)
            results.setdefault("face_contrast_raw", None)
            results.setdefault("face_mean_brightness", None)
            results.setdefault("face_noise_raw", None)

        else:
            for m in face_metrics:
                results.setdefault(f"face_{m}_score", default_score.get(m, 0.0))
                # 直書き "ok" でも良いが SSOT を使う
                results.setdefault(f"face_{m}_eval_status", STATUS_OK)

    def _normalize_for_csv(self, results: Dict[str, Any]) -> None:
        """
        CSVに書く前の正規化。
        faces などの複合オブジェクトは「Python repr」ではなく「JSON文字列」で保存する。

        これにより後段で json.loads() が安定し、
        "faces が str のまま" という型不一致を防ぎやすくなる。
        """
        import json

        def _to_json_str(v) -> str:
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                # 最後の砦: それでもダメなら文字列化（ただしJSONでない可能性あり）
                return str(v)

        faces = results.get("faces", [])
        if isinstance(faces, (list, dict)):
            results["faces"] = _to_json_str(faces)
        else:
            # すでに文字列なら、そのままJSONとして解釈できる可能性があるので保持
            # （ただしPython reprが来る可能性もある）
            results["faces"] = str(faces)

        gaze = results.get("gaze")
        if isinstance(gaze, (list, dict)):
            results["gaze"] = _to_json_str(gaze)

    def _calc_lead_room_score(self, image: np.ndarray, best_face: Dict[str, Any], yaw: float) -> float:
        h, W = image.shape[:2]
        box = best_face.get("box") or best_face.get("bbox")
        if not box or len(box) != 4:
            return 0.0

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0

        left_space = cx
        right_space = W - cx

        if yaw > 5:
            raw = (right_space - left_space) / W
        elif yaw < -5:
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

        base = 0.6 * eye_contact + 0.4 * face_direction
        yaw_penalty = min(yaw / 45.0, 1.0)
        pitch_penalty = min(pitch / 45.0, 1.0)
        penalty = 0.5 * yaw_penalty + 0.5 * pitch_penalty
        score = base * (1.0 - 0.4 * penalty)

        score = max(0.0, min(1.0, score))

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

    def _try_eval(self, evaluator, image: np.ndarray, name: str) -> Dict[str, Any]:
        try:
            return evaluator.evaluate(image) or {}
        except Exception as e:
            self.logger.error(f"{name} evaluate failed: {e}")
            self.logger.error(traceback.format_exc())
            return {}

    def _load_evaluator_config(self) -> Dict[str, Any]:
        """
        Evaluator の discretize 閾値やノイズ等の評価パラメータを読む。
        例: config/evaluator_thresholds.yaml

        config_manager があればそこからパスを取得。
        無ければデフォルトパスを読む。読めない場合は {} を返す。
        """
        path = None
        if self.config_manager is not None:
            try:
                path = self.config_manager.get(
                    "evaluator_thresholds_path",
                    default="config/evaluator_thresholds.yaml",
                )
            except Exception:
                path = "config/evaluator_thresholds.yaml"
        else:
            path = "config/evaluator_thresholds.yaml"

        try:
            if not path or not isinstance(path, str):
                return {}
            if not os.path.exists(path):
                self.logger.warning(f"Evaluator config not found: {path} (use defaults)")
                return {}

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            if not isinstance(data, dict):
                self.logger.warning(f"Evaluator config is not a dict: {path} (use defaults)")
                return {}

            self.logger.info(f"Evaluator config loaded: {path}")
            return data

        except Exception as e:
            self.logger.warning(f"Evaluator config load failed: {e} (use defaults)")
            return {}

    @staticmethod
    def decide_accept_static(
        results: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        return decide_accept(results, thresholds=thresholds)
