from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _is_nan(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return x != x


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if _is_nan(value):
            return default
        return float(value)

    text = str(value).strip()
    if text == "":
        return default

    try:
        parsed = float(text)
    except Exception:
        return default

    if _is_nan(parsed):
        return default
    return parsed


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if _is_nan(value):
            return default
        return value != 0

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off", ""}:
        return False
    return default


def serialize_reason_list(reasons: list[str]) -> str:
    return "|".join(reason for reason in reasons if reason)


@dataclass(frozen=True)
class KeepReasonThresholds:
    # SNS受け
    expression_score: float = 0.75
    composition_score: float = 0.75
    eye_contact_score: float = 0.75
    face_position_score: float = 0.75

    # モデル受け
    face_exposure_score: float = 0.75
    face_sharpness_score: float = 0.75
    pose_score: float = 75.0

    # プロ目線で強い
    pro_composition_strong: float = 0.85
    pro_face_composition_strong: float = 0.85


@dataclass(frozen=True)
class GreenDecisionThresholds:
    version: str = "v2-real"

    # 最低成立
    require_face_detected: bool = True
    min_blurriness_score: float = 0.50
    min_composition_score: float = 0.50

    # 補助条件
    min_face_exposure_score: float = 0.50
    reject_no_face_shot_type: bool = True

    keep_reason_thresholds: KeepReasonThresholds = field(default_factory=KeepReasonThresholds)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "GreenDecisionThresholds":
        if not config:
            return cls()

        raw = config.get("green_decision", config)
        keep_raw = raw.get("keep_reason_thresholds", {})

        return cls(
            version=str(raw.get("version", "v2-real")),
            require_face_detected=bool(raw.get("require_face_detected", True)),
            min_blurriness_score=_to_float(raw.get("min_blurriness_score", 0.50), 0.50),
            min_composition_score=_to_float(raw.get("min_composition_score", 0.50), 0.50),
            min_face_exposure_score=_to_float(
                raw.get("min_face_exposure_score", 0.50),
                0.50,
            ),
            reject_no_face_shot_type=bool(raw.get("reject_no_face_shot_type", True)),
            keep_reason_thresholds=KeepReasonThresholds(
                expression_score=_to_float(keep_raw.get("expression_score", 0.75), 0.75),
                composition_score=_to_float(keep_raw.get("composition_score", 0.75), 0.75),
                eye_contact_score=_to_float(keep_raw.get("eye_contact_score", 0.75), 0.75),
                face_position_score=_to_float(keep_raw.get("face_position_score", 0.75), 0.75),
                face_exposure_score=_to_float(keep_raw.get("face_exposure_score", 0.75), 0.75),
                face_sharpness_score=_to_float(keep_raw.get("face_sharpness_score", 0.75), 0.75),
                pose_score=_to_float(keep_raw.get("pose_score", 75.0), 75.0),
                pro_composition_strong=_to_float(
                    keep_raw.get("pro_composition_strong", 0.85),
                    0.85,
                ),
                pro_face_composition_strong=_to_float(
                    keep_raw.get("pro_face_composition_strong", 0.85),
                    0.85,
                ),
            ),
        )


@dataclass(frozen=True)
class GreenDecision:
    is_green: bool
    minimum_pass: bool
    keep_reasons: list[str]
    reject_reasons: list[str]

    def to_row_updates(self, version: str) -> dict[str, Any]:
        return {
            "is_green": self.is_green,
            "green_minimum_pass": self.minimum_pass,
            "green_keep_reasons": serialize_reason_list(self.keep_reasons),
            "green_reject_reasons": serialize_reason_list(self.reject_reasons),
            "green_decision_version": version,
        }


def is_green_minimum_pass(
    row: Mapping[str, Any],
    thresholds: GreenDecisionThresholds,
) -> tuple[bool, list[str]]:
    reject_reasons: list[str] = []

    shot_type = str(row.get("shot_type", "") or "").strip().lower()
    face_detected = _to_bool(row.get("face_detected", False), False)
    blurriness_score = _to_float(row.get("blurriness_score", 0.0), 0.0)
    composition_score = _to_float(row.get("composition_score", 0.0), 0.0)
    face_exposure_score = _to_float(row.get("face_exposure_score", 0.0), 0.0)

    if thresholds.reject_no_face_shot_type and shot_type == "no_face":
        reject_reasons.append("shot_type_no_face")

    if thresholds.require_face_detected and not face_detected:
        reject_reasons.append("face_not_detected")

    if blurriness_score < thresholds.min_blurriness_score:
        reject_reasons.append("blurriness_too_low")

    if composition_score < thresholds.min_composition_score:
        reject_reasons.append("composition_too_low")

    if face_detected and face_exposure_score < thresholds.min_face_exposure_score:
        reject_reasons.append("face_exposure_too_low")

    return (len(reject_reasons) == 0, reject_reasons)


def collect_green_keep_reasons(
    row: Mapping[str, Any],
    thresholds: GreenDecisionThresholds,
) -> list[str]:
    keep_reasons: list[str] = []
    t = thresholds.keep_reason_thresholds

    expression_score = _to_float(row.get("expression_score", 0.0), 0.0)
    composition_score = _to_float(row.get("composition_score", 0.0), 0.0)
    eye_contact_score = _to_float(row.get("eye_contact_score", 0.0), 0.0)
    face_position_score = _to_float(row.get("face_position_score", 0.0), 0.0)

    face_exposure_score = _to_float(row.get("face_exposure_score", 0.0), 0.0)
    face_sharpness_score = _to_float(row.get("face_sharpness_score", 0.0), 0.0)
    pose_score = _to_float(row.get("pose_score", 0.0), 0.0)

    face_composition_score = _to_float(row.get("face_composition_score", 0.0), 0.0)

    # SNS受け
    if expression_score >= t.expression_score:
        keep_reasons.append("sns_expression")

    if composition_score >= t.composition_score:
        keep_reasons.append("sns_composition")

    if eye_contact_score >= t.eye_contact_score:
        keep_reasons.append("sns_eye_contact")

    if face_position_score >= t.face_position_score:
        keep_reasons.append("sns_face_position")

    # モデル受け
    if face_exposure_score >= t.face_exposure_score:
        keep_reasons.append("model_face_exposure")

    if face_sharpness_score >= t.face_sharpness_score:
        keep_reasons.append("model_face_sharpness")

    if pose_score >= t.pose_score:
        keep_reasons.append("model_pose")

    # プロ目線
    if composition_score >= t.pro_composition_strong:
        keep_reasons.append("pro_composition_strong")

    if face_composition_score >= t.pro_face_composition_strong:
        keep_reasons.append("pro_face_composition_strong")

    # 順序維持で重複排除
    seen: set[str] = set()
    deduped: list[str] = []
    for reason in keep_reasons:
        if reason not in seen:
            seen.add(reason)
            deduped.append(reason)

    return deduped


def decide_green(
    row: Mapping[str, Any],
    thresholds: GreenDecisionThresholds | Mapping[str, Any] | None = None,
) -> GreenDecision:
    resolved_thresholds = (
        thresholds
        if isinstance(thresholds, GreenDecisionThresholds)
        else GreenDecisionThresholds.from_config(thresholds)
    )

    minimum_pass, reject_reasons = is_green_minimum_pass(
        row=row,
        thresholds=resolved_thresholds,
    )

    if not minimum_pass:
        return GreenDecision(
            is_green=False,
            minimum_pass=False,
            keep_reasons=[],
            reject_reasons=reject_reasons,
        )

    keep_reasons = collect_green_keep_reasons(
        row=row,
        thresholds=resolved_thresholds,
    )

    final_reject_reasons = list(reject_reasons)
    if not keep_reasons:
        final_reject_reasons.append("no_keep_reason")

    return GreenDecision(
        is_green=(len(keep_reasons) > 0),
        minimum_pass=True,
        keep_reasons=keep_reasons,
        reject_reasons=final_reject_reasons,
    )


def apply_green_decision_to_row(
    row: Mapping[str, Any],
    thresholds: GreenDecisionThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_thresholds = (
        thresholds
        if isinstance(thresholds, GreenDecisionThresholds)
        else GreenDecisionThresholds.from_config(thresholds)
    )
    decision = decide_green(row=row, thresholds=resolved_thresholds)
    return decision.to_row_updates(version=resolved_thresholds.version)


__all__ = [
    "GreenDecision",
    "GreenDecisionThresholds",
    "KeepReasonThresholds",
    "apply_green_decision_to_row",
    "collect_green_keep_reasons",
    "decide_green",
    "is_green_minimum_pass",
    "serialize_reason_list",
]
