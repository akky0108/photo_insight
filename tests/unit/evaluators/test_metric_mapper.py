import pytest

from photo_insight.evaluators.portrait_quality.metric_mapping import MetricResultMapper


def test_mapper_noise_global():
    """
    global noise_raw が生成されるか
    """
    mapper = MetricResultMapper()

    result = {
        "noise_score": 0.75,
        "noise_sigma_used": 0.01,
        "noise_eval_status": "ok",
    }

    out = mapper.map("noise", result, prefix="")

    assert "noise_raw" in out
    assert out["noise_raw"] == pytest.approx(-0.01)


def test_mapper_noise_face():
    """
    face_noise_raw が生成されるか
    """
    mapper = MetricResultMapper()

    result = {
        "noise_score": 0.5,
        "noise_sigma_used": 0.02,
        "noise_eval_status": "ok",
    }

    out = mapper.map("noise", result, prefix="face_")

    assert "face_noise_raw" in out
    assert out["face_noise_raw"] == pytest.approx(-0.02)


def test_mapper_noise_missing_sigma():
    """
    sigma無しでも落ちないか
    """
    mapper = MetricResultMapper()

    result = {
        "noise_score": 0.5,
    }

    out = mapper.map("noise", result, prefix="")

    assert "noise_raw" in out
    assert out["noise_raw"] is None
