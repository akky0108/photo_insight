from __future__ import annotations

from photo_insight.pipelines.evaluation_rank._internal.green_decision import (
    GreenDecisionThresholds,
    apply_green_decision_to_row,
    decide_green,
)


def _base_row(**overrides):
    row = {
        "file_name": "test.NEF",
        "shot_type": "face_only",
        "face_detected": True,
        "blurriness_score": 0.75,
        "composition_score": 0.70,
        "face_exposure_score": 0.70,
        "face_sharpness_score": 0.70,
        "face_composition_score": 0.70,
        "expression_score": 0.50,
        "eye_contact_score": 0.50,
        "face_position_score": 0.50,
        "pose_score": 50.0,
    }
    row.update(overrides)
    return row


def test_decide_green_rejects_no_face_shot_type():
    row = _base_row(
        shot_type="no_face",
        face_detected=False,
    )

    decision = decide_green(row)

    assert decision.minimum_pass is False
    assert decision.is_green is False
    assert "shot_type_no_face" in decision.reject_reasons


def test_decide_green_rejects_when_face_not_detected():
    row = _base_row(face_detected=False)

    decision = decide_green(row)

    assert decision.minimum_pass is False
    assert decision.is_green is False
    assert "face_not_detected" in decision.reject_reasons


def test_decide_green_rejects_when_minimum_pass_ok_but_no_keep_reason():
    row = _base_row(
        expression_score=0.50,
        composition_score=0.60,
        eye_contact_score=0.40,
        face_position_score=0.40,
        face_exposure_score=0.60,
        face_sharpness_score=0.60,
        face_composition_score=0.60,
        pose_score=60.0,
    )

    decision = decide_green(row)

    assert decision.minimum_pass is True
    assert decision.is_green is False
    assert decision.keep_reasons == []
    assert "no_keep_reason" in decision.reject_reasons


def test_decide_green_accepts_with_sns_expression_reason():
    row = _base_row(
        expression_score=0.90,
        composition_score=0.70,
    )

    decision = decide_green(row)

    assert decision.minimum_pass is True
    assert decision.is_green is True
    assert "sns_expression" in decision.keep_reasons


def test_decide_green_accepts_with_model_face_exposure_reason():
    row = _base_row(
        face_exposure_score=0.90,
    )

    decision = decide_green(row)

    assert decision.minimum_pass is True
    assert decision.is_green is True
    assert "model_face_exposure" in decision.keep_reasons


def test_decide_green_collects_multiple_keep_reasons():
    row = _base_row(
        expression_score=0.90,
        eye_contact_score=0.80,
        face_exposure_score=0.92,
        face_sharpness_score=0.88,
        face_composition_score=0.90,
        composition_score=0.87,
        pose_score=88.0,
    )

    decision = decide_green(row)

    assert decision.minimum_pass is True
    assert decision.is_green is True
    assert "sns_expression" in decision.keep_reasons
    assert "sns_eye_contact" in decision.keep_reasons
    assert "model_face_exposure" in decision.keep_reasons
    assert "model_face_sharpness" in decision.keep_reasons
    assert "pro_composition_strong" in decision.keep_reasons
    assert "pro_face_composition_strong" in decision.keep_reasons
    assert "model_pose" in decision.keep_reasons


def test_apply_green_decision_to_row_returns_expected_fields():
    row = {
        "shot_type": "portrait",
        "face_detected": True,
        "blurriness_score": 0.8,
        "composition_score": 0.8,
        "face_exposure_score": 0.8,
        "expression_score": 0.9,
    }

    result = apply_green_decision_to_row(row)

    assert set(result.keys()) == {
        "is_green",
        "green_hard_gate_pass",
        "green_hard_gate_reasons",
        "green_minimum_pass",
        "green_keep_reasons",
        "green_reject_reasons",
        "green_decision_version",
    }
    assert "green_hard_gate_pass" in result
    assert "green_hard_gate_reasons" in result


def test_decide_green_respects_custom_thresholds():
    thresholds = GreenDecisionThresholds.from_config(
        {
            "green_decision": {
                "min_blurriness_score": 0.80,
                "min_composition_score": 0.80,
                "keep_reason_thresholds": {
                    "expression_score": 0.95,
                },
            }
        }
    )

    row = _base_row(
        blurriness_score=0.75,
        composition_score=0.70,
        expression_score=0.90,
    )

    decision = decide_green(row, thresholds)

    assert decision.minimum_pass is False
    assert decision.is_green is False
    assert "blurriness_too_low" in decision.reject_reasons
    assert "composition_too_low" in decision.reject_reasons
