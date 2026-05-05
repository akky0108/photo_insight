from photo_insight.pipelines.evaluation_rank.criteria7015 import (
    evaluate_criteria7015,
)
from photo_insight.pipelines.evaluation_rank.distribution_health import (
    DistributionHealthMetrics,
)


def test_criteria7015_pass():
    metrics = DistributionHealthMetrics(
        total_count=100,
        accepted_count=50,
        accepted_ratio=0.5,
        saturation_ratio=0.05,
        discrete_ratio=0.1,
    )

    result = evaluate_criteria7015(metrics)

    assert result.validation_pass is True
    assert result.reasons == []


def test_criteria7015_fail_accepted_ratio():
    metrics = DistributionHealthMetrics(
        total_count=100,
        accepted_count=95,
        accepted_ratio=0.95,
        saturation_ratio=0.05,
        discrete_ratio=0.1,
    )

    result = evaluate_criteria7015(metrics)

    assert result.validation_pass is False
    assert any("accepted_ratio" in r for r in result.reasons)


def test_criteria7015_fail_saturation():
    metrics = DistributionHealthMetrics(
        total_count=100,
        accepted_count=50,
        accepted_ratio=0.5,
        saturation_ratio=0.3,
        discrete_ratio=0.1,
    )

    result = evaluate_criteria7015(metrics)

    assert result.validation_pass is False
    assert any("saturation_ratio" in r for r in result.reasons)


def test_criteria7015_fail_discrete():
    metrics = DistributionHealthMetrics(
        total_count=100,
        accepted_count=50,
        accepted_ratio=0.5,
        saturation_ratio=0.05,
        discrete_ratio=0.8,
    )

    result = evaluate_criteria7015(metrics)

    assert result.validation_pass is False
    assert any("discrete_ratio" in r for r in result.reasons)
