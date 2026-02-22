# src/evaluators/composite_composition_evaluator.py
# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from photo_insight.evaluators.rule_based_composition_evaluator import (
    RuleBasedCompositionEvaluator,
)
from photo_insight.evaluators.fullbody_composition_evaluator import (
    FullBodyCompositionEvaluator,
)
from photo_insight.evaluators.base_composition_evaluator import BaseCompositionEvaluator
from photo_insight.evaluators.rule_of_thirds_evaluator import (
    RuleOfThirdsEvaluator,
)  # ★ 追加
from photo_insight.utils.app_logger import Logger


class CompositeCompositionEvaluator(BaseCompositionEvaluator):
    """
    顔構図評価と全身構図評価、およびルール・オブ・サード構図評価を統合するファサードクラス。

    追加（観測性/契約強化）:
      - composition_eval_status: ok / fallback_used / invalid
      - composition_invalid_reason: invalid/fallback の理由（ok は空）
      - main_subject_center_status: ok / invalid / none
      - main_subject_center_invalid_reason: center の invalid 理由（ok/none は空）

    注意:
      - composition_status は従来の「統合状態（face_only等）」を維持する
      - main_subject_center_source は source 専用（invalid,face_box 等の混ぜ表現は禁止）
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger = logger or Logger(logger_name="CompositeCompositionEvaluator")

        self.config = config or {}
        comp_conf = self.config.get("composition", {})

        self.face_weight: float = float(comp_conf.get("face_weight", 1.0))
        self.body_weight: float = float(comp_conf.get("body_weight", 1.0))
        self.rule_of_thirds_weight: float = float(
            comp_conf.get("rule_of_thirds_weight", 0.5)
        )

        disc_conf = comp_conf.get("discretize_thresholds", {})
        self.threshold_excellent: float = float(disc_conf.get("excellent", 0.85))
        self.threshold_good: float = float(disc_conf.get("good", 0.70))
        self.threshold_fair: float = float(disc_conf.get("fair", 0.55))
        self.threshold_poor: float = float(disc_conf.get("poor", 0.40))

        self.face_evaluator = RuleBasedCompositionEvaluator(
            logger=self.logger,
            config=self.config,
        )
        self.body_evaluator = FullBodyCompositionEvaluator(
            logger=self.logger,
            config=self.config,
        )

    def evaluate(
        self,
        image: np.ndarray,
        face_boxes: List[Dict[str, Any]],
        body_keypoints: List[Optional[List[float]]],
    ) -> Dict[str, Any]:
        """
        顔と全身の構図評価、およびルール・オブ・サード評価を統合して実行。

        Returns:
            dict: 統合された評価結果。
        """
        self.logger.info("Starting composite composition evaluation.")

        # -------------------------
        # basic input validation (観測性のため早期に理由を固定)
        # -------------------------
        if not isinstance(image, np.ndarray):
            return self._invalid_result(
                reason="invalid_input:image_type_not_ndarray",
                face_results={},
                body_results={},
            )

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return self._invalid_result(
                reason="invalid_input:image_size_invalid",
                face_results={},
                body_results={},
            )

        # face_boxes / body_keypoints の型を守る（CSV由来の壊れた入力をここで捕捉）
        if face_boxes is None:
            face_boxes = []
        if not isinstance(face_boxes, list):
            return self._invalid_result(
                reason=(
                    f"invalid_input:face_boxes_type_mismatch:"
                    f"{type(face_boxes).__name__}"
                ),
                face_results={},
                body_results={},
            )
        if any((fb is not None and not isinstance(fb, dict)) for fb in face_boxes):
            return self._invalid_result(
                reason="invalid_input:face_boxes_element_type_mismatch",
                face_results={},
                body_results={},
            )

        if body_keypoints is None:
            body_keypoints = []
        if not isinstance(body_keypoints, list):
            return self._invalid_result(
                reason=(
                    f"invalid_input:body_keypoints_type_mismatch:"
                    f"{type(body_keypoints).__name__}"
                ),
                face_results={},
                body_results={},
            )

        # -------------------------
        # run underlying evaluators
        # -------------------------
        face_results: Dict[str, Any] = (
            self.face_evaluator.evaluate(image, face_boxes) or {}
        )
        body_results: Dict[str, Any] = (
            self.body_evaluator.evaluate(image, body_keypoints) or {}
        )

        face_raw = face_results.get("face_composition_raw")
        body_raw = body_results.get("body_composition_raw")

        # -------------------------
        # RoT: center & score (+ reason)
        # -------------------------
        (
            rule_of_thirds_raw,
            main_subject_center,
            main_subject_center_source,
            main_subject_center_status,
            main_subject_center_invalid_reason,
        ) = self._compute_rule_of_thirds_raw(
            image=image,
            face_results=face_results,
            body_results=body_results,
            face_boxes=face_boxes,
            body_keypoints=body_keypoints,
        )

        # -------------------------
        # merge
        # -------------------------
        composition_raw, status = self._merge_raw_scores(
            face_raw=face_raw,
            body_raw=body_raw,
            rule_of_thirds_raw=rule_of_thirds_raw,
        )

        # composition_score + eval_status
        composition_invalid_reason = ""
        if composition_raw is None:
            # 「分からないのでニュートラル」運用（fallback）
            composition_score = 0.5
            if status == "not_computed":
                status = "not_computed_with_default"
            composition_eval_status = "fallback_used"
            composition_invalid_reason = status  # fallback 理由として status を残す
        else:
            composition_score = self._to_discrete_score(composition_raw)
            composition_eval_status = "ok"

        # face/body discrete score補完
        face_score = face_results.get("face_composition_score")
        if face_score is None and face_raw is not None:
            face_score = self._to_discrete_score(face_raw)

        body_score = body_results.get("body_composition_score")
        if body_score is None and body_raw is not None:
            body_score = self._to_discrete_score(body_raw)

        # RoT discrete score
        rule_of_thirds_score = (
            self._to_discrete_score(rule_of_thirds_raw)
            if rule_of_thirds_raw is not None
            else None
        )

        # contrib
        contrib_rule_of_thirds = (
            self.rule_of_thirds_weight * float(rule_of_thirds_raw)
            if rule_of_thirds_raw is not None
            else None
        )

        contrib_body_composition = (
            self.body_weight * float(body_raw)
            if body_raw is not None and self.body_weight > 0.0
            else None
        )

        # center expansion
        if main_subject_center is not None:
            try:
                center_x, center_y = main_subject_center
                main_subject_center_x = float(center_x)
                main_subject_center_y = float(center_y)
            except Exception:
                main_subject_center_x = None
                main_subject_center_y = None
                # ここは“内部矛盾”なので center 側を invalid 扱いに倒す
                if main_subject_center_status == "ok":
                    main_subject_center_status = "invalid"
                    main_subject_center_invalid_reason = "center_unpack_failed"
        else:
            main_subject_center_x = None
            main_subject_center_y = None

        combined_results: Dict[str, Any] = {
            **face_results,
            **body_results,  # 注意: group_id / subgroup_id は全身側が上書きする
            "face_composition_raw": face_raw,
            "face_composition_score": face_score,
            "body_composition_raw": body_raw,
            "body_composition_score": body_score,
            "rule_of_thirds_raw": rule_of_thirds_raw,
            "rule_of_thirds_score": rule_of_thirds_score,
            "contrib_comp_rule_of_thirds_score": contrib_rule_of_thirds,
            "contrib_comp_body_composition_score": contrib_body_composition,
            "main_subject_center_source": main_subject_center_source,
            "main_subject_center_status": main_subject_center_status,
            "main_subject_center_invalid_reason": main_subject_center_invalid_reason,
            "main_subject_center_x": main_subject_center_x,
            "main_subject_center_y": main_subject_center_y,
            "composition_raw": composition_raw,
            "composition_score": composition_score,
            "composition_status": status,
            "composition_eval_status": composition_eval_status,
            "composition_invalid_reason": composition_invalid_reason,
        }

        # -------------------------
        # logging pack（観測性）
        # -------------------------
        self.logger.info(
            "Composite composition evaluation completed. "
            f"composition_raw={composition_raw}, "
            f"composition_score={composition_score}, "
            f"composition_status={status}, "
            f"composition_eval_status={composition_eval_status}, "
            f"composition_invalid_reason={composition_invalid_reason}, "
            f"rot_raw={rule_of_thirds_raw}, center={main_subject_center}, "
            f"center_source={main_subject_center_source}, "
            f"center_status={main_subject_center_status}, "
            f"center_invalid_reason={main_subject_center_invalid_reason}, "
            f"face_boxes_n={len(face_boxes)}, "
            f"body_keypoints_n={len(body_keypoints)}, "
            f"img_w={w}, img_h={h}"
        )

        return combined_results

    # -----------------------
    # helper: invalid unified return
    # -----------------------
    def _invalid_result(
        self,
        reason: str,
        face_results: Dict[str, Any],
        body_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        # 最小限の返却（既存列は壊さない）
        return {
            **(face_results or {}),
            **(body_results or {}),
            "composition_raw": None,
            "composition_score": 0.5,
            "composition_status": "invalid",
            "composition_eval_status": "invalid",
            "composition_invalid_reason": reason,
            "rule_of_thirds_raw": None,
            "rule_of_thirds_score": None,
            "main_subject_center_source": None,
            "main_subject_center_status": "invalid",
            "main_subject_center_invalid_reason": reason,
            "main_subject_center_x": None,
            "main_subject_center_y": None,
        }

    # -----------------------
    # merge / discretize
    # -----------------------
    def _merge_raw_scores(
        self,
        face_raw: Optional[float],
        body_raw: Optional[float],
        rule_of_thirds_raw: Optional[float] = None,
    ) -> Tuple[Optional[float], str]:
        if rule_of_thirds_raw is None or self.rule_of_thirds_weight <= 0.0:
            if face_raw is None and body_raw is None:
                return None, "not_computed"

            if face_raw is not None and (body_raw is None or body_raw <= 0.0):
                return float(face_raw), "face_only"

            if body_raw is not None and (face_raw is None or face_raw <= 0.0):
                return float(body_raw), "body_only"

            w_face = self.face_weight
            w_body = self.body_weight
            if w_face <= 0 and w_body <= 0:
                raw = (float(face_raw) + float(body_raw)) / 2.0
                return raw, "both_zero_weight"

            raw = (float(face_raw) * w_face + float(body_raw) * w_body) / (
                w_face + w_body
            )
            return raw, "face_and_body"

        has_face = face_raw is not None and face_raw > 0.0
        has_body = body_raw is not None and body_raw > 0.0
        has_rot = rule_of_thirds_raw is not None and rule_of_thirds_raw > 0.0

        if not has_face and not has_body and not has_rot:
            return None, "not_computed"

        total = 0.0
        wsum = 0.0
        parts: List[str] = []

        if has_face and self.face_weight > 0.0:
            total += float(face_raw) * self.face_weight
            wsum += self.face_weight
            parts.append("face")

        if has_body and self.body_weight > 0.0:
            total += float(body_raw) * self.body_weight
            wsum += self.body_weight
            parts.append("body")

        if has_rot and self.rule_of_thirds_weight > 0.0:
            total += float(rule_of_thirds_raw) * self.rule_of_thirds_weight
            wsum += self.rule_of_thirds_weight
            parts.append("rule_of_thirds")

        if wsum <= 0.0:
            vals = [
                float(v)
                for v in [face_raw, body_raw, rule_of_thirds_raw]
                if v is not None
            ]
            if not vals:
                return None, "not_computed"
            return float(sum(vals) / len(vals)), "fallback_average"

        raw = total / wsum

        if parts == ["face"]:
            status = "face_only"
        elif parts == ["body"]:
            status = "body_only"
        elif parts == ["rule_of_thirds"]:
            status = "rule_of_thirds_only"
        else:
            status = "_and_".join(parts)

        return raw, status

    def _to_discrete_score(self, value: float) -> float:
        if value is None:
            return 0.5

        v = float(value)

        if v >= self.threshold_excellent:
            return 1.0
        if v >= self.threshold_good:
            return 0.75
        if v >= self.threshold_fair:
            return 0.5
        if v >= self.threshold_poor:
            return 0.25
        return 0.0

    # -----------------------
    # Rule of Thirds center picking (+ reason)
    # -----------------------
    def _compute_rule_of_thirds_raw(
        self,
        image: np.ndarray,
        face_results: Dict[str, Any],
        body_results: Dict[str, Any],
        face_boxes: List[Dict[str, Any]],
        body_keypoints: List[Optional[List[float]]],
    ) -> Tuple[
        Optional[float],
        Optional[Tuple[float, float]],
        Optional[str],
        str,
        str,
    ]:
        """
        Returns:
            (rule_of_thirds_raw, main_subject_center, main_subject_center_source,
             main_subject_center_status, main_subject_center_invalid_reason)
        """
        evaluator: Optional[RuleOfThirdsEvaluator] = None

        def _eval_with_center(
            center: Any, source: str
        ) -> Tuple[
            Optional[float], Optional[Tuple[float, float]], Optional[str], str, str
        ]:
            nonlocal evaluator
            ok, norm_center, invalid_reason = self._validate_and_normalize_center(
                center, image
            )
            if not ok:
                return None, None, source, "invalid", invalid_reason
            if evaluator is None:
                evaluator = RuleOfThirdsEvaluator(image)
            score = evaluator.evaluate_from_point(norm_center)
            if score is None or (not np.isfinite(float(score))):
                return None, None, source, "invalid", "rot_score_nonfinite"
            return float(score), norm_center, source, "ok", ""

        # 1) face_results["main_subject_center"]
        if "main_subject_center" in face_results:
            score, c, s, st, r = _eval_with_center(
                face_results.get("main_subject_center"), "main_subject_center"
            )
            if score is not None:
                return score, c, s, st, r

        # 2) face_results["face_center"]
        if "face_center" in face_results:
            score, c, s, st, r = _eval_with_center(
                face_results.get("face_center"), "face_center"
            )
            if score is not None:
                return score, c, s, st, r

        # 3) body_results["full_body_center"]
        if "full_body_center" in body_results:
            score, c, s, st, r = _eval_with_center(
                body_results.get("full_body_center"), "full_body_center"
            )
            if score is not None:
                return score, c, s, st, r

        # 4) face_boxes[0]
        if face_boxes:
            cx, cy, reason = self._center_from_face_box_with_reason(
                face_boxes[0], image
            )
            if cx is not None and cy is not None:
                score, c, s, st, r = _eval_with_center((cx, cy), "face_box")
                if score is not None:
                    return score, c, s, st, r
            # face_box があるのに取れない場合は理由を保持
            last_reason = f"face_box_failed:{reason}"
        else:
            last_reason = "face_boxes_missing"

        # 5) body_keypoints[0]
        if body_keypoints:
            cx, cy, reason = self._center_from_body_keypoints_with_reason(
                body_keypoints[0], image
            )
            if cx is not None and cy is not None:
                score, c, s, st, r = _eval_with_center((cx, cy), "body_keypoints")
                if score is not None:
                    return score, c, s, st, r
            last_reason = f"body_keypoints_failed:{reason}"

        # 6) none
        self.logger.debug(
            "Rule-of-thirds center could not be determined; skipping RoT contribution. "
            f"reason={last_reason}"
        )
        return None, None, None, "none", last_reason

    @staticmethod
    def _validate_and_normalize_center(
        center: Any, image: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[float, float]], str]:
        """
        center を (x,y) に解釈し、画像境界に収める。
        - ok: True のとき norm_center は (clipped_x, clipped_y)
        - 失敗理由は invalid_reason として返す
        """
        if center is None:
            return False, None, "center_missing"

        try:
            x, y = center
        except Exception:
            return False, None, "center_type_mismatch"

        try:
            x = float(x)
            y = float(y)
        except Exception:
            return False, None, "center_cast_failed"

        if not (np.isfinite(x) and np.isfinite(y)):
            return False, None, "center_nonfinite"

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return False, None, "img_size_invalid"

        # 許容: 画像内（境界含む）
        if x < 0.0 or x > float(w) or y < 0.0 or y > float(h):
            return False, None, "center_out_of_range"

        # ここでは clamp 不要だが、浮動小数の微小誤差対策でclip
        cx = float(np.clip(x, 0.0, float(w)))
        cy = float(np.clip(y, 0.0, float(h)))
        return True, (cx, cy), ""

    @staticmethod
    def _center_from_face_box_with_reason(
        box: Dict[str, Any],
        image: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        顔検出 bbox から中心を推定する（理由付き）。

        Returns:
            (cx, cy, reason)  where reason is "" on success.
        """
        if box is None or not isinstance(box, dict):
            return None, None, "face_box_type_mismatch"

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return None, None, "img_size_invalid"

        # helper: finalize
        def _finalize(
            cx: float, cy: float
        ) -> Tuple[Optional[float], Optional[float], str]:
            if not (np.isfinite(cx) and np.isfinite(cy)):
                return None, None, "face_box_center_nonfinite"
            return (
                float(np.clip(cx, 0.0, float(w))),
                float(np.clip(cy, 0.0, float(h))),
                "",
            )

        try:
            # pattern1: x,y,w,h  (xywh)
            if all(k in box for k in ("x", "y", "w", "h")):
                x = float(box["x"])
                y = float(box["y"])
                bw = float(box["w"])
                bh = float(box["h"])
                if bw <= 0.0 or bh <= 0.0:
                    return None, None, "face_box_degenerate_xywh"
                return _finalize(x + bw / 2.0, y + bh / 2.0)

            # pattern2: x_min,y_min,x_max,y_max (xyxy)
            if all(k in box for k in ("x_min", "y_min", "x_max", "y_max")):
                x_min = float(box["x_min"])
                y_min = float(box["y_min"])
                x_max = float(box["x_max"])
                y_max = float(box["y_max"])
                if x_max <= x_min or y_max <= y_min:
                    return None, None, "face_box_degenerate_xyxy_keys"
                return _finalize((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

            # pattern3: bbox: [x_min,y_min,x_max,y_max]
            if (
                "bbox" in box
                and isinstance(box["bbox"], (list, tuple))
                and len(box["bbox"]) == 4
            ):
                x_min, y_min, x_max, y_max = map(float, box["bbox"])
                if x_max <= x_min or y_max <= y_min:
                    return None, None, "face_box_degenerate_bbox"
                return _finalize((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

            # pattern4: box: [x_min,y_min,x_max,y_max]
            if (
                "box" in box
                and isinstance(box["box"], (list, tuple))
                and len(box["box"]) == 4
            ):
                x_min, y_min, x_max, y_max = map(float, box["box"])
                if x_max <= x_min or y_max <= y_min:
                    return None, None, "face_box_degenerate_box"
                return _finalize((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

        except Exception as e:
            return None, None, f"face_box_parse_failed:{type(e).__name__}"

        return None, None, "face_box_invalid_format"

    @staticmethod
    def _center_from_body_keypoints_with_reason(
        keypoints: Optional[List[float]],
        image: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        全身キーポイントから重心を推定（理由付き）。
        """
        if not keypoints:
            return None, None, "body_keypoints_missing"

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return None, None, "img_size_invalid"

        try:
            arr = np.asarray(keypoints, dtype=float)
        except Exception as e:
            return None, None, f"body_keypoints_cast_failed:{type(e).__name__}"

        if arr.ndim != 1 or arr.size < 2:
            return None, None, "body_keypoints_invalid_shape"

        xs = arr[0::2]
        ys = arr[1::2]
        if xs.size == 0 or ys.size == 0:
            return None, None, "body_keypoints_empty"

        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            return None, None, "body_keypoints_nonfinite"

        cx = float(xs.mean())
        cy = float(ys.mean())

        if not (np.isfinite(cx) and np.isfinite(cy)):
            return None, None, "body_keypoints_center_nonfinite"

        cx = float(np.clip(cx, 0.0, float(w)))
        cy = float(np.clip(cy, 0.0, float(h)))
        return cx, cy, ""
