import pytest
from evaluators.portrait_quality.portrait_quality_evaluator import PortraitQualityEvaluator


def _base(**overrides):
    r = dict(
        # face
        face_detected=True,
        yaw=0,
        exposure_score=1.0,
        face_sharpness_score=0,
        face_noise_score=0,
        delta_face_sharpness=-999,

        # composition
        composition_rule_based_score=0,
        framing_score=0,
        lead_room_score=0.0,

        # technical
        noise_score=0,
        contrast_score=0,
        blurriness_score=0.0,

        # full body (NEW)
        full_body_detected=False,
        pose_score=0.0,
        full_body_cut_risk=1.0,
    )
    r.update(overrides)
    return r


def _decide(results):
    return PortraitQualityEvaluator.decide_accept_static(results)


def test_no_face():
    accepted, reason = _decide(_base(face_detected=False))
    assert accepted is False
    assert reason == "no_face"


def test_full_body_accept():
    accepted, reason = _decide(_base(
        face_detected=False,
        full_body_detected=True,
        pose_score=55,
        full_body_cut_risk=0.6,
        noise_score=60,
        blurriness_score=0.45,
        exposure_score=0.5,
    ))
    assert accepted is True
    assert reason == "full_body"


def test_full_body_rejected():
    accepted, reason = _decide(_base(
        face_detected=False,
        full_body_detected=True,
        pose_score=10,              # 条件未達
        full_body_cut_risk=0.9,     # 条件未達
        noise_score=10,
        blurriness_score=0.1,
        exposure_score=0.0,
    ))
    assert accepted is False
    assert reason == "full_body_rejected"


def test_full_body_rejected_by_cut_risk():
    accepted, reason = _decide(_base(
        face_detected=False,
        full_body_detected=True,
        pose_score=80,           # ここは満たす
        full_body_cut_risk=0.9,  # ここで落ちる
        noise_score=80,
        blurriness_score=0.6,
        exposure_score=1.0,
    ))
    assert accepted is False
    assert reason == "full_body_rejected"


def test_face_quality():
    accepted, reason = _decide(_base(
        exposure_score=0.5,
        face_sharpness_score=75,
        face_noise_score=70,
        contrast_score=55,
        blurriness_score=0.55,
        delta_face_sharpness=-10,
        yaw=30,
    ))
    assert accepted is True
    assert reason == "face_quality"


def test_composition():
    accepted, reason = _decide(_base(
        exposure_score=0.5,
        composition_rule_based_score=70,
        framing_score=60,
        lead_room_score=0.10,
        noise_score=60,
        blurriness_score=0.45,
    ))
    assert accepted is True
    assert reason == "composition"


def test_technical():
    accepted, reason = _decide(_base(
        exposure_score=1.0,
        noise_score=70,
        contrast_score=60,
        blurriness_score=0.60,
        delta_face_sharpness=-15,
    ))
    assert accepted is True
    assert reason == "technical"


def test_rejected():
    accepted, reason = _decide(_base(
        exposure_score=0.0,
        face_sharpness_score=10,
        face_noise_score=10,
        contrast_score=10,
        blurriness_score=0.1,
        delta_face_sharpness=-999,
        yaw=90,
    ))
    assert accepted is False
    assert reason == "rejected"


def test_priority_face_quality_over_composition():
    accepted, reason = _decide(_base(
        # face_quality 条件
        exposure_score=0.5,
        face_sharpness_score=75,
        face_noise_score=70,
        contrast_score=60,
        blurriness_score=0.60,
        delta_face_sharpness=-10,
        yaw=10,

        # composition 条件も満たす
        composition_rule_based_score=80,
        framing_score=80,
        lead_room_score=0.20,
        noise_score=80,
    ))
    assert accepted is True
    assert reason == "face_quality"


def test_priority_composition_over_technical():
    accepted, reason = _decide(_base(
        # composition 条件
        exposure_score=1.0,
        composition_rule_based_score=70,
        framing_score=60,
        lead_room_score=0.10,
        noise_score=70,
        blurriness_score=0.60,

        # technical 条件も満たす
        contrast_score=60,
        delta_face_sharpness=-15,
    ))
    assert accepted is True
    assert reason == "composition"


def test_full_body_does_not_override_face_routes():
    accepted, reason = _decide(_base(
        face_detected=True,
        full_body_detected=True,
        pose_score=100,
        full_body_cut_risk=0.0,
        # face_quality で落ちる条件にする
        exposure_score=0.0,
        face_sharpness_score=0,
        face_noise_score=0,
        blurriness_score=0.0,
        contrast_score=0,
    ))
    assert accepted is False
    assert reason == "rejected"

    