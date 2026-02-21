from photo_insight.evaluators.portrait_quality.portrait_quality_evaluator import (
    PortraitQualityEvaluator,
)


def _base(**overrides):
    r = dict(
        # face
        face_detected=True,
        yaw=0,
        exposure_score=1.0,
        face_sharpness_score=0,
        face_noise_score=0,
        delta_face_sharpness=-999,
        composition_rule_based_score=0,
        framing_score=0,
        lead_room_score=0.0,
        noise_score=0,
        contrast_score=0,
        blurriness_score=0.0,
    )
    r.update(overrides)
    return r


def _decide(results):
    # thresholds=None で呼ぶので、decide_accept_static のデフォルト値を使う
    return PortraitQualityEvaluator.decide_accept_static(results)


def test_no_face():
    accepted, reason = _decide(_base(face_detected=False))
    assert accepted is False
    assert reason == "no_face"


def test_face_quality():
    accepted, reason = _decide(
        _base(
            exposure_score=0.5,
            face_sharpness_score=0.75,  # 0〜1
            face_noise_score=0.8,  # good 以上
            contrast_score=55,
            blurriness_score=0.55,
            delta_face_sharpness=-10,
            yaw=30,
        )
    )
    assert accepted is True
    assert reason == "face_quality"


def test_composition():
    accepted, reason = _decide(
        _base(
            exposure_score=0.5,
            composition_rule_based_score=70,  # 0〜100 の旧スコア（フォールバック経由）
            framing_score=0.6,
            lead_room_score=0.10,
            noise_score=0.6,
            blurriness_score=0.45,
        )
    )
    assert accepted is True
    assert reason == "composition"


def test_technical():
    accepted, reason = _decide(
        _base(
            exposure_score=1.0,
            noise_score=0.8,
            contrast_score=0.60,
            blurriness_score=0.60,
            face_sharpness_score=0.5,
            face_blurriness_score=0.60,
            delta_face_sharpness=-15,
        )
    )
    assert accepted is True, f"rejected: {reason}"
    assert reason == "technical"


def test_rejected():
    accepted, reason = _decide(
        _base(
            exposure_score=0.0,
            face_sharpness_score=0.1,
            face_noise_score=0.1,
            contrast_score=10,
            blurriness_score=0.1,
            delta_face_sharpness=-999,
            yaw=90,
        )
    )
    assert accepted is False
    assert reason == "rejected"


def test_priority_face_quality_over_composition():
    accepted, reason = _decide(
        _base(
            # face_quality 条件
            exposure_score=0.5,
            face_sharpness_score=0.8,
            face_noise_score=0.8,
            contrast_score=60,
            blurriness_score=0.60,
            delta_face_sharpness=-10,
            yaw=10,
            # composition 条件も満たす
            composition_rule_based_score=80,
            framing_score=0.8,
            lead_room_score=0.20,
            noise_score=0.8,
        )
    )
    assert accepted is True
    assert reason == "face_quality"


def test_priority_composition_over_technical():
    accepted, reason = _decide(
        _base(
            # composition 条件
            exposure_score=1.0,
            composition_rule_based_score=70,
            framing_score=0.6,
            lead_room_score=0.10,
            noise_score=0.7,
            blurriness_score=0.60,
            # technical 条件も満たす
            contrast_score=60,
            delta_face_sharpness=-15,
        )
    )
    assert accepted is True
    assert reason == "composition"
