from __future__ import annotations

from typing import Any, Dict

import pytest

# 現行構成に合わせて import fallback を用意
try:
    from photo_insight.pipelines.evaluation_rank.acceptance import (
        AcceptanceEngine,
        AcceptRules,
    )
except ImportError:
    from batch_processor.evaluation_rank.acceptance import (  # type: ignore
        AcceptanceEngine,
        AcceptRules,
    )


Row = Dict[str, Any]


def _make_engine(
    *,
    face_sharpness_min: float = 0.60,
    reject_on_missing_face_sharpness: bool = True,
) -> AcceptanceEngine:
    """
    テスト用に選定条件をできるだけ単純化した engine を作る。
    単一行でも Green 判定が走るように ratio を 1.0 に寄せる。
    """
    rules = AcceptRules(
        portrait_percentile=70.0,
        non_face_percentile=80.0,
        portrait_secondary_percentile=60.0,
        non_face_secondary_percentile=70.0,
        green_ratio_total=1.0,
        green_min_total=0,
        green_ratio_small=1.0,
        green_ratio_mid=1.0,
        green_ratio_large=1.0,
        green_count_small_max=60,
        green_count_mid_max=120,
        green_per_group_enabled=False,
        green_per_group_min_each=0,
        flag_ratio=1.0,
        flag_min_total=0,
        eye_patch_min=70,
        eye_half_min=0.85,
        eye_closed_min=0.98,
        face_portrait_required=True,
        face_sharpness_min=face_sharpness_min,
        reject_on_missing_face_sharpness=reject_on_missing_face_sharpness,
    )
    return AcceptanceEngine(rules=rules)


def _base_row(**overrides: Any) -> Row:
    """
    acceptance.py が参照するキーを最低限そろえたベース行。
    """
    row: Row = {
        "file_name": "test.jpg",
        "face_detected": True,
        "shot_type": "close_up",
        "group_id": "A",
        "subgroup_id": "1",
        "overall_score": 88.0,
        "score_face": 82.0,
        "score_composition": 70.0,
        "score_technical": 75.0,
        "face_sharpness_score": 0.80,
        "eye_closed_prob_best": None,
        "eye_patch_size_best": None,
        "eye_contact_score": 0.90,
        "face_direction_score": 0.88,
        "framing_score": 0.86,
        "face_position_score": 0.87,
        "composition_rule_based_score": 0.84,
        "rule_of_thirds_score": 0.82,
        "lead_room_score": 0.75,
        "body_composition_score": 0.70,
        "expression_score": 0.80,
        "debug_expression": 0.80,
        "debug_expr_effective": 0.80,
        "debug_half_penalty": 0.0,
        "accepted_flag": 0,
        "secondary_accept_flag": 0,
        "accepted_reason": "",
        "secondary_accept_reason": "",
        "face_portrait_candidate": True,
    }
    row.update(overrides)
    return row


def test_face_not_detected_portrait_candidate_is_forced_reject() -> None:
    engine = _make_engine()

    row = _base_row(
        file_name="no_face_portrait.jpg",
        face_detected=False,
        face_portrait_candidate=True,
        shot_type="close_up",
        face_sharpness_score=0.95,
    )

    rows = [row]
    thresholds = engine.run(rows)

    assert "portrait" in thresholds
    assert row["category"] == "portrait"
    assert row["accepted_flag"] == 0
    assert row["secondary_accept_flag"] == 0
    assert "FACE_NOT_DETECTED" in str(row["accepted_reason"])


def test_low_face_sharpness_portrait_is_forced_reject() -> None:
    engine = _make_engine(face_sharpness_min=0.60)

    row = _base_row(
        file_name="blur_face_portrait.jpg",
        face_detected=True,
        shot_type="close_up",
        face_sharpness_score=0.42,
    )

    rows = [row]
    engine.run(rows)

    assert row["category"] == "portrait"
    assert row["accepted_flag"] == 0
    assert row["secondary_accept_flag"] == 0
    assert "FACE_SHARPNESS_LOW" in str(row["accepted_reason"])


def test_missing_face_sharpness_portrait_is_not_force_rejected_when_value_is_missing() -> None:
    engine = _make_engine(
        face_sharpness_min=0.60,
        reject_on_missing_face_sharpness=True,
    )

    row = _base_row(
        file_name="missing_face_sharpness.jpg",
        face_detected=True,
        shot_type="close_up",
        face_sharpness_score=None,
    )

    rows = [row]
    engine.run(rows)

    assert row["category"] == "portrait"
    assert "FACE_SHARPNESS_MISSING" not in str(row["accepted_reason"])
    assert "FACE_SHARPNESS_LOW" not in str(row["accepted_reason"])


def test_good_face_portrait_remains_green() -> None:
    engine = _make_engine(face_sharpness_min=0.60)

    row = _base_row(
        file_name="good_face_portrait.jpg",
        face_detected=True,
        shot_type="close_up",
        face_sharpness_score=0.91,
        overall_score=90.0,
        score_face=85.0,
        score_composition=72.0,
        score_technical=78.0,
    )

    rows = [row]
    engine.run(rows)

    assert row["category"] == "portrait"
    assert row["accepted_flag"] == 1
    assert row["secondary_accept_flag"] == 0
    assert "FACE_NOT_DETECTED" not in str(row["accepted_reason"])
    assert "FACE_SHARPNESS_LOW" not in str(row["accepted_reason"])
    assert "FACE_SHARPNESS_MISSING" not in str(row["accepted_reason"])


def test_non_face_without_portrait_hint_is_not_target_of_hard_reject() -> None:
    engine = _make_engine()

    row = _base_row(
        file_name="landscape.jpg",
        face_detected=False,
        face_portrait_candidate=False,
        shot_type="",
        face_sharpness_score=None,
        overall_score=92.0,
        score_face=0.0,
        score_composition=80.0,
        score_technical=82.0,
    )

    rows = [row]
    engine.run(rows)

    assert row["category"] == "non_face"
    assert "FACE_NOT_DETECTED" not in str(row["accepted_reason"])
    assert "FACE_SHARPNESS_MISSING" not in str(row["accepted_reason"])


@pytest.mark.parametrize(
    ("sharpness", "expected_reason"),
    [
        ("0.10", "FACE_SHARPNESS_LOW"),
        ("0.59", "FACE_SHARPNESS_LOW"),
        ("bad-value", "FACE_SHARPNESS_INVALID"),
    ],
)
def test_face_sharpness_edge_cases(sharpness: Any, expected_reason: str) -> None:
    engine = _make_engine(face_sharpness_min=0.60)

    row = _base_row(
        file_name="edge_case.jpg",
        face_detected=True,
        shot_type="close_up",
        face_sharpness_score=sharpness,
    )

    rows = [row]
    engine.run(rows)

    assert row["accepted_flag"] == 0
    assert row["secondary_accept_flag"] == 0
    assert expected_reason in str(row["accepted_reason"])
