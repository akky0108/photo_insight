# src/evaluators/portrait_quality/portrait_quality_evaluator.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import traceback
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

import cv2
import numpy as np
import yaml

from photo_insight.core.logging import Logger
from photo_insight.image_utils.image_preprocessor import ImagePreprocessor

from photo_insight.evaluators.quality_thresholds import QualityThresholds
from photo_insight.evaluators.portrait_accept_rules import decide_accept
from photo_insight.evaluators.portrait_quality.metric_mapping import MetricResultMapper
from photo_insight.evaluators.portrait_quality.category import (
    classify_portrait_category,
    to_legacy_category,
)

from photo_insight.evaluators.common.grade_contract import (
    STATUS_NOT_COMPUTED,
    STATUS_FALLBACK,
    STATUS_OK,
    normalize_eval_status,
)

# NOTE:
# - heavy deps を含む可能性がある内部モジュールは「ここで import しない」。
# - __init__ 内または専用メソッドで遅延 import する（CI の import 時点で落とさない）。
if TYPE_CHECKING:  # pragma: no cover
    from photo_insight.face_detectors.face_processor import FaceProcessor


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

    方針
    ----
    - CI/テストで heavy deps（insightface 等）が無い場合でも
      「import しただけ」で落ちない設計
    - 実運用では必要な依存が入っていれば通常動作
    - skip_face_processing=True の場合は顔関連の import/初期化を一切行わない
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
        face_processor: Optional["FaceProcessor"] = None,
        skip_face_processing: bool = False,
        config_manager: Optional[Any] = None,
        quality_profile: str = "portrait",
        thresholds_path: Optional[str] = None,
    ):
        self.is_raw = is_raw
        self.logger = logger or Logger(logger_name="PortraitQualityEvaluator")
        self.config_manager = config_manager
        self.file_name = file_name if isinstance(image_input, np.ndarray) else os.path.basename(image_input)
        self.image_path = image_input if isinstance(image_input, str) else None
        self.skip_face_processing = bool(skip_face_processing)

        # 現状は未使用だが、既存インターフェース互換のため保持
        self.local_region_size = int(local_region_size)
        self.preprocessor_resize_size = tuple(preprocessor_resize_size)

        # --- evaluator 用の閾値/設定（discretize_thresholds_raw 等）をロード ---
        self.eval_config = self._load_evaluator_config()

        # -------------------------
        # Preprocess (image IO / resize)
        # -------------------------
        self.preprocessor = ImagePreprocessor(
            logger=self.logger,
            is_raw=self.is_raw,
            gamma=1.2,
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

        # 評価用 uint8（恒久対策）
        self.rgb_u8 = images["original_u8"]
        self.resized_2048_bgr_u8 = images["resized_2048_bgr_u8"]
        self.bgr_u8 = images["original_bgr_u8"]

        img = self.resized_2048_bgr_u8
        self.logger.info(f"[debug] eval_img dtype={img.dtype}, min={img.min()}, max={img.max()}, mean={img.mean():.2f}")

        # -------------------------
        # Face (optional / lazy)
        # -------------------------
        self.face_evaluator = None
        self.face_processor: Optional["FaceProcessor"] = None

        if not self.skip_face_processing:
            self.face_evaluator, self.face_processor = self._build_face_stack(
                face_processor=face_processor,
            )

        # -------------------------
        # Evaluators (global / face)
        # -------------------------
        (
            SharpnessEvaluator,
            BlurrinessEvaluator,
            ContrastEvaluator,
            NoiseEvaluator,
            ExposureEvaluator,
            LocalSharpnessEvaluator,
            LocalContrastEvaluator,
            ColorBalanceEvaluator,
            CompositeCompositionEvaluator,
        ) = self._lazy_import_evaluators_stack()

        self.evaluators_global = {
            "face": self.face_evaluator,
            "sharpness": SharpnessEvaluator(
                logger=self.logger,
                config=self.eval_config,
            ),
            "blurriness": BlurrinessEvaluator(
                logger=self.logger,
                config=self.eval_config,
            ),
            "contrast": ContrastEvaluator(
                logger=self.logger,
                config=self.eval_config,
                metric_key="contrast",
            ),
            "noise": NoiseEvaluator(
                max_noise_value=max_noise_value,
                logger=self.logger,
                config=self.eval_config,
            ),
            "local_sharpness": LocalSharpnessEvaluator(
                logger=self.logger,
                config=self.eval_config,
            ),
            "local_contrast": LocalContrastEvaluator(),
            "exposure": ExposureEvaluator(),
            "color_balance": ColorBalanceEvaluator(),
        }

        self.evaluators_face = {
            "face": self.face_evaluator,
            "sharpness": SharpnessEvaluator(
                logger=self.logger,
                config=self._subcfg("face_sharpness"),
            ),
            "blurriness": BlurrinessEvaluator(
                logger=self.logger,
                config=self._subcfg("face_blurriness"),
            ),
            "contrast": ContrastEvaluator(
                logger=self.logger,
                config=self._subcfg("face_contrast"),
                metric_key="contrast",
            ),
            "noise": NoiseEvaluator(
                max_noise_value=max_noise_value,
                logger=self.logger,
                config=self._subcfg("face_noise"),
            ),
            "local_sharpness": LocalSharpnessEvaluator(
                logger=self.logger,
                config=self._subcfg("face_local_sharpness"),
            ),
            "local_contrast": LocalContrastEvaluator(),
            "exposure": ExposureEvaluator(),
            "color_balance": ColorBalanceEvaluator(),
        }

        # -------------------------
        # Body detector (optional / lazy)
        # -------------------------
        self.body_detector = self._try_make_body_detector()

        # composition evaluator (Composite)
        self.composition_evaluator = CompositeCompositionEvaluator(
            logger=self.logger,
            config=None,
        )

        self.mapper = MetricResultMapper()

        self.logger.info(f"画像ファイル {self.file_name} をロードしました")

        # 閾値系の初期化
        self._init_thresholds(
            quality_profile=quality_profile,
            thresholds_path=thresholds_path,
        )

    @staticmethod
    def decide_accept_static(
        results: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[bool, str]:
        return decide_accept(results, thresholds=thresholds)

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
            elif self.face_processor is None or self.face_evaluator is None:
                self.logger.warning("顔処理スタックが未初期化のため、顔処理をスキップします。")
            elif hasattr(self.face_evaluator, "available") and not self.face_evaluator.available():
                self.logger.warning("face_evaluator が利用不可のため、顔処理をスキップします。")
            else:
                face_result = self.face_processor.detect_faces(self.rgb_u8) or {"faces": []}

            faces = face_result.get("faces", []) or []
            results["face_detected"] = bool(faces)
            results["faces"] = faces

            # -------------------------
            # full body / pose
            # -------------------------
            body_result: Dict[str, Any]
            if self.body_detector is None:
                body_result = {
                    "full_body_detected": False,
                    "pose_score": 0.0,
                    "full_body_cut_risk": 1.0,
                    "keypoints": [],
                }
            else:
                try:
                    body_result = self.body_detector.detect(self.bgr_u8) or {}
                except AttributeError:
                    # legacy fallback
                    full_body = bool(self.body_detector.detect_full_body(self.bgr_u8))
                    body_result = {
                        "full_body_detected": full_body,
                        "pose_score": 100.0 if full_body else 0.0,
                        "full_body_cut_risk": 0.0 if full_body else 1.0,
                        "keypoints": [],
                    }
                except Exception as e:
                    self.logger.warning(f"全身検出に失敗: {e}")
                    body_result = {
                        "full_body_detected": False,
                        "pose_score": 0.0,
                        "full_body_cut_risk": 1.0,
                        "keypoints": [],
                    }

            results.update(body_result)
            body_keypoints = body_result.get("keypoints") or body_result.get("body_keypoints") or []

            # -------------------------
            # composition
            # -------------------------
            results.update(self._evaluate_composition(self.rgb_u8, faces, body_keypoints))

            # -------------------------
            # face portrait candidate
            # -------------------------
            results["face_portrait_candidate"] = self._detect_face_portrait_candidate(
                face_detected=bool(faces),
                face_boxes=faces,
                results=results,
            )

            # -------------------------
            # face attributes + face region metrics
            # -------------------------
            best_face = None
            if (
                (not self.skip_face_processing)
                and self.face_processor is not None
                and self.face_evaluator is not None
                and (not hasattr(self.face_evaluator, "available") or self.face_evaluator.available())
            ):
                best_face = self.face_processor.get_best_face(faces)

            if best_face and self.face_processor is not None:
                cropped_face_rgb_u8 = self.face_processor.crop_face(self.rgb_u8, best_face)
                if cropped_face_rgb_u8 is not None:
                    cropped_face_bgr_u8 = cv2.cvtColor(cropped_face_rgb_u8, cv2.COLOR_RGB2BGR)

                    face_attrs = self.face_processor.extract_attributes(best_face) or {}
                    results["yaw"] = face_attrs.get("yaw", 0.0)
                    results["pitch"] = face_attrs.get("pitch", 0.0)
                    results["roll"] = face_attrs.get("roll", 0.0)
                    results["gaze"] = face_attrs.get("gaze", [0, 0])

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
                        self.rgb_u8,
                        best_face,
                        float(results.get("yaw", 0.0)),
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

            accepted, reason = self.decide_accept_static(
                results,
                thresholds=self.decision_thresholds,
            )
            results["accepted_flag"] = accepted
            results["accepted_reason"] = reason

            # -------------------------
            # finalize (schema + csv normalization)
            # -------------------------
            self._ensure_result_schema(results)
            self._add_portrait_category(results)
            self._normalize_for_csv(results)

            return results

        except Exception as e:
            self.logger.error(f"評価中のエラー: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}

    # ============================================================
    # build / runtime config
    # ============================================================
    def _build_face_stack(
        self,
        *,
        face_processor: Optional["FaceProcessor"] = None,
    ):
        """
        FaceEvaluator / FaceProcessor を遅延 import し、初期化して返す。

        Notes
        -----
        - insightface 実行設定は config_manager から取得する
        - FaceEvaluator 側へ runtime config をそのまま渡す
        - face_processor が DI 済みならそれを優先使用する
        - strict=True の場合は fail-fast、False の場合は安全にフォールバックする

        Parameters
        ----------
        face_processor : Optional["FaceProcessor"]
            既存の FaceProcessor を外部から注入する場合に使用する

        Returns
        -------
        tuple[Any, Optional["FaceProcessor"]]
            (face_evaluator, face_processor)
        """
        runtime_cfg = self._get_insightface_runtime_config()

        model_name = str(runtime_cfg.get("model_name", "buffalo_l"))
        model_root = str(runtime_cfg.get("model_root", "/work/models/insightface"))

        providers = runtime_cfg.get("providers", ["CPUExecutionProvider"])
        if not providers:
            providers = ["CPUExecutionProvider"]
        providers = list(providers)

        det_size_raw = runtime_cfg.get("det_size", [640, 640])
        if not isinstance(det_size_raw, (list, tuple)) or len(det_size_raw) != 2:
            det_size = (640, 640)
        else:
            det_size = (int(det_size_raw[0]), int(det_size_raw[1]))

        strict = bool(runtime_cfg.get("strict", True))

        # テストや軽量実行環境で config_manager が無い場合は、
        # heavy dependency 未導入でも portrait_quality 全体は継続できるようにする。
        if self.config_manager is None:
            strict = False

        self.logger.info(
            "InsightFace runtime config: "
            f"model_name={model_name}, "
            f"model_root={model_root}, "
            f"providers={providers}, "
            f"det_size={det_size}, "
            f"strict={strict}"
        )

        try:
            FaceEvaluator, FaceProcessor = self._lazy_import_face_stack()

            evaluator = FaceEvaluator(
                backend="insightface",
                confidence_threshold=0.5,
                gpu=False,
                strict=strict,
                model_name=model_name,
                model_root=model_root,
                providers=providers,
                det_size=det_size,
            )

            if hasattr(evaluator, "available") and not evaluator.available():
                msg = "FaceEvaluator is initialized but unavailable. " "顔処理をフォールバックします。"
                if strict:
                    raise RuntimeError(msg)
                self.logger.warning(msg)
                return evaluator, None

            processor = face_processor or FaceProcessor(evaluator, logger=self.logger)
            return evaluator, processor

        except Exception as e:
            if strict:
                raise
            self.logger.warning(f"Face stack build failed (fallback to no-face processing). detail={e}")
            return None, None

    def _get_insightface_runtime_config(self) -> Dict[str, Any]:
        """
        insightface 実行設定を config_manager から取得する。
        """
        default_cfg: Dict[str, Any] = {
            "model_name": "buffalo_l",
            "model_root": "/work/models/insightface",
            "providers": ["CPUExecutionProvider"],
            "det_size": [640, 640],
            "strict": True,
        }

        if self.config_manager is None:
            return default_cfg

        try:
            raw = self.config_manager.get("portrait_quality.insightface", default=None)
            if isinstance(raw, dict):
                cfg = dict(default_cfg)
                cfg.update(raw)

                providers = cfg.get("providers")
                if not providers:
                    cfg["providers"] = ["CPUExecutionProvider"]

                det_size = cfg.get("det_size", [640, 640])
                if not isinstance(det_size, (list, tuple)) or len(det_size) != 2:
                    cfg["det_size"] = [640, 640]

                return cfg
        except Exception as e:
            self.logger.warning(f"insightface runtime config load failed: {e}")

        return default_cfg

    def _subcfg(self, metric_key: str) -> Dict[str, Any]:
        """
        face_* 系の設定を、各 evaluator が期待する base key に寄せて返す。

        例
        ----
        metric_key="face_blurriness" -> {"blurriness": ...}
        """
        cfg = self.eval_config if isinstance(self.eval_config, dict) else {}
        spec = cfg.get(metric_key)
        if not isinstance(spec, dict):
            return {}

        base = metric_key.replace("face_", "")
        return {base: spec}

    # ============================================================
    # lazy imports
    # ============================================================
    def _lazy_import_face_stack(self):
        """
        FaceEvaluator / FaceProcessor は insightface 等に依存し得るので遅延 import。
        """
        try:
            from photo_insight.evaluators.face_evaluator import FaceEvaluator
            from photo_insight.face_detectors.face_processor import FaceProcessor
        except Exception as e:
            raise RuntimeError(
                "Face stack (FaceEvaluator/FaceProcessor) import failed. "
                "insightface 等の依存が未導入の可能性があります。"
                "テスト/CIでは skip_face_processing=True を使用するか、依存を導入してください。"
                f" (detail: {e})"
            ) from e

        return FaceEvaluator, FaceProcessor

    def _lazy_import_evaluators_stack(self):
        """
        評価器群は内部で追加依存を持つ可能性があるため遅延 import。
        """
        from photo_insight.evaluators.sharpness_evaluator import SharpnessEvaluator
        from photo_insight.evaluators.blurriness_evaluator import BlurrinessEvaluator
        from photo_insight.evaluators.contrast_evaluator import ContrastEvaluator
        from photo_insight.evaluators.noise_evaluator import NoiseEvaluator
        from photo_insight.evaluators.exposure_evaluator import ExposureEvaluator
        from photo_insight.evaluators.local_sharpness_evaluator import (
            LocalSharpnessEvaluator,
        )
        from photo_insight.evaluators.local_contrast_evaluator import (
            LocalContrastEvaluator,
        )
        from photo_insight.evaluators.color_balance_evaluator import (
            ColorBalanceEvaluator,
        )
        from photo_insight.evaluators.composite_composition_evaluator import (
            CompositeCompositionEvaluator,
        )

        return (
            SharpnessEvaluator,
            BlurrinessEvaluator,
            ContrastEvaluator,
            NoiseEvaluator,
            ExposureEvaluator,
            LocalSharpnessEvaluator,
            LocalContrastEvaluator,
            ColorBalanceEvaluator,
            CompositeCompositionEvaluator,
        )

    def _try_make_body_detector(self):
        """
        FullBodyDetector は mediapipe 等に依存し得るため optional。
        無ければ evaluate() で安全な fallback を返す。
        """
        try:
            from photo_insight.detectors.body_detection import FullBodyDetector

            return FullBodyDetector()
        except Exception as e:
            self.logger.warning(f"FullBodyDetector import/init failed (fallback to no-body). detail={e}")
            return None

    # ============================================================
    # composition
    # ============================================================
    def _evaluate_composition(
        self,
        image: np.ndarray,
        face_boxes: list,
        body_keypoints: list,
    ) -> Dict[str, Any]:
        def _empty_payload(status: str) -> Dict[str, Any]:
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
            if not face_boxes and not body_keypoints:
                self.logger.warning("構図評価スキップ：face_boxes も body_keypoints も空です。")
                return _empty_payload(status=STATUS_NOT_COMPUTED)

            result = (
                self.composition_evaluator.evaluate(
                    image=image,
                    face_boxes=face_boxes,
                    body_keypoints=body_keypoints,
                )
                or {}
            )
            self.logger.info(f"構図評価結果: {result}")

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

    # ============================================================
    # metric eval helpers
    # ============================================================
    def _eval_metrics(
        self,
        image: np.ndarray,
        metrics,
        prefix: str,
        tag: str,
        evaluators_dict,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in metrics:
            evaluator = evaluators_dict.get(name)
            if evaluator is None:
                continue
            result = self._try_eval(evaluator, image, name=f"{tag}:{name}")
            out.update(self.mapper.map(name, result, prefix=prefix))
        return out

    def _try_eval(self, evaluator, image: np.ndarray, name: str) -> Dict[str, Any]:
        try:
            return evaluator.evaluate(image) or {}
        except Exception as e:
            self.logger.error(f"{name} evaluate failed: {e}")
            self.logger.error(traceback.format_exc())
            return {}

    # ============================================================
    # schema & normalize
    # ============================================================
    def _ensure_result_schema(self, results: Dict[str, Any]) -> None:
        results.setdefault("face_detected", False)
        results.setdefault("face_portrait_candidate", False)
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

            for metric_name in face_metrics:
                results.setdefault(
                    f"face_{metric_name}_score",
                    default_score.get(metric_name, 0.0),
                )
                results[f"face_{metric_name}_eval_status"] = status
                results.setdefault(f"face_{metric_name}_fallback_reason", reason)

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
            for metric_name in face_metrics:
                results.setdefault(
                    f"face_{metric_name}_score",
                    default_score.get(metric_name, 0.0),
                )
                results.setdefault(f"face_{metric_name}_eval_status", STATUS_OK)

    def _add_portrait_category(self, results: Dict[str, Any]) -> None:
        """
        3分類 portrait_category を results に追加する。

        NOTE:
        - accepted_flag / accepted_reason の判定後に呼ぶことで、既存の採否判定には影響させない。
        - 既存 evaluation_rank 互換のため、category が未設定の場合のみ portrait/non_face を補完する。
        - category を portrait_face / portrait_body に置き換えない。
        """
        portrait_category = classify_portrait_category(
            face_detected=bool(results.get("face_detected")),
            face_portrait_candidate=bool(results.get("face_portrait_candidate")),
            full_body_detected=bool(results.get("full_body_detected")),
            shot_type=(results.get("shot_type") or ""),
        )

        results["portrait_category"] = portrait_category.value

        if not results.get("category"):
            results["category"] = to_legacy_category(portrait_category)

    def _detect_face_portrait_candidate(
        self,
        *,
        face_detected: bool,
        face_boxes: list,
        results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        顔ポートレート候補判定。

        - face_detected=True なら candidate=True
        - face_detected=False でも、構図的に顔主体の portrait らしい場合は True

        現段階では誤爆を避けるため、近接〜上半身構図のみを弱いヒントとして使う。
        """
        if face_detected:
            return True

        r = results or {}

        shot_type = str(r.get("shot_type") or "").strip().lower()
        full_body_detected = bool(r.get("full_body_detected"))

        close_portrait_types = {"face_only", "close_up", "upper_body"}

        if shot_type in close_portrait_types and not full_body_detected:
            return True

        return False

    def _normalize_for_csv(self, results: Dict[str, Any]) -> None:
        def _to_json_str(value: Any) -> str:
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)

        faces = results.get("faces", [])

        # results["faces"] は構造体のまま維持（unitテスト/呼び出し側の契約）
        if isinstance(faces, (list, dict)):
            results["faces_json"] = _to_json_str(faces)
        else:
            results["faces_json"] = str(faces)

        gaze = results.get("gaze")
        if isinstance(gaze, (list, dict)):
            results["gaze"] = _to_json_str(gaze)

    # ============================================================
    # derived
    # ============================================================
    def _calc_lead_room_score(
        self,
        image: np.ndarray,
        best_face: Dict[str, Any],
        yaw: float,
    ) -> float:
        _h, width = image.shape[:2]
        box = best_face.get("box") or best_face.get("bbox")
        if not box or len(box) != 4:
            return 0.0

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0

        left_space = cx
        right_space = width - cx

        if yaw > 5:
            raw = (right_space - left_space) / width
        elif yaw < -5:
            raw = (left_space - right_space) / width
        else:
            raw = 0.0

        return float(max(-1.0, min(1.0, raw)))

    def _add_face_global_deltas(self, results: Dict[str, Any]) -> None:
        def _diff(a: Any, b: Any):
            if a is None or b is None:
                return None
            try:
                return float(a) - float(b)
            except (TypeError, ValueError):
                return None

        results["delta_face_sharpness"] = _diff(
            results.get("face_sharpness_score"),
            results.get("sharpness_score"),
        )
        results["delta_face_contrast"] = _diff(
            results.get("face_contrast_score"),
            results.get("contrast_score"),
        )

    def _add_expression_score(self, results: Dict[str, Any]) -> None:
        def _f(key: str, default: float = 0.0) -> float:
            value = results.get(key, default)
            try:
                return float(value)
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

    # ============================================================
    # threshold / config
    # ============================================================
    def _init_thresholds(
        self,
        quality_profile: str,
        thresholds_path: Optional[str],
    ) -> None:
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
            "noise_ok": float(decision.get("noise_ok", 0.5)),
            "noise_good": float(decision.get("noise_good", 0.75)),
            "face_noise_good": float(decision.get("face_noise_good", 0.75)),
            "full_body_body_height_min": float(decision.get("full_body_body_height_min", 0.30)),
            "full_body_cut_risk_max_for_shot_type": float(decision.get("full_body_cut_risk_max_for_shot_type", 0.90)),
            "full_body_footroom_min_for_shot_type": float(decision.get("full_body_footroom_min_for_shot_type", 0.00)),
            "seated_center_y_min": float(decision.get("seated_center_y_min", 0.50)),
            "seated_center_y_max": float(decision.get("seated_center_y_max", 0.75)),
            "seated_footroom_max": float(decision.get("seated_footroom_max", 0.22)),
            "seated_body_height_min": float(decision.get("seated_body_height_min", 0.30)),
            "upper_body_headroom_min": float(decision.get("upper_body_headroom_min", 0.15)),
            "upper_body_footroom_max": float(decision.get("upper_body_footroom_max", 0.15)),
            "face_only_body_height_max": float(decision.get("face_only_body_height_max", 0.35)),
            "center_side_margin_min": float(decision.get("center_side_margin_min", 0.02)),
            "full_body_face_noise_min": float(decision.get("full_body_face_noise_min", 0.5)),
            "full_body_pose_min_100": float(decision.get("full_body_pose_min_100", 55.0)),
            "full_body_cut_risk_max": float(decision.get("full_body_cut_risk_max", 0.6)),
            "blurriness_min_full_body": float(decision.get("blurriness_min_full_body", 0.45)),
            "exposure_min_common": float(decision.get("exposure_min_common", 0.5)),
            "full_body_footroom_min": float(decision.get("full_body_footroom_min", 0.0)),
            "full_body_contrast_min": float(decision.get("full_body_contrast_min", 0.40)),
            "full_body_face_contrast_min": float(decision.get("full_body_face_contrast_min", 0.50)),
            "composition_score_min": float(decision.get("composition_score_min", 0.75)),
            "framing_score_min": float(decision.get("framing_score_min", 0.5)),
            "lead_room_score_min": float(decision.get("lead_room_score_min", 0.10)),
            "face_quality_exposure_min": float(decision.get("face_quality_exposure_min", 0.5)),
            "face_quality_face_sharpness_min": float(decision.get("face_quality_face_sharpness_min", 0.75)),
            "face_quality_contrast_min": float(decision.get("face_quality_contrast_min", 0.55)),
            "face_quality_blur_min": float(decision.get("face_quality_blur_min", 0.55)),
            "face_quality_delta_face_sharpness_min": float(
                decision.get("face_quality_delta_face_sharpness_min", -10.0)
            ),
            "face_quality_yaw_max_abs_deg": float(decision.get("face_quality_yaw_max_abs_deg", 30.0)),
            "expression_min": float(decision.get("expression_min", 0.5)),
            "technical_contrast_min": float(decision.get("technical_contrast_min", 0.60)),
            "technical_blur_min": float(decision.get("technical_blur_min", 0.60)),
            "technical_delta_face_sharpness_min": float(decision.get("technical_delta_face_sharpness_min", -15.0)),
            "technical_exposure_min": float(decision.get("technical_exposure_min", 1.0)),
        }

    def _load_evaluator_config(self) -> Dict[str, Any]:
        path: Optional[str]
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

            with open(path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}

            if not isinstance(data, dict):
                self.logger.warning(f"Evaluator config is not a dict: {path} (use defaults)")
                return {}

            self.logger.info(f"Evaluator config loaded: {path}")
            return data

        except Exception as e:
            self.logger.warning(f"Evaluator config load failed: {e} (use defaults)")
            return {}
