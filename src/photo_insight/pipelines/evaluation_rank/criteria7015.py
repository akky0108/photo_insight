from __future__ import annotations

from dataclasses import dataclass

from .distribution_health import DistributionHealthMetrics


@dataclass(frozen=True)
class Criteria7015Result:
    """Criteria7015 の判定結果。"""

    validation_pass: bool
    reasons: list[str]


@dataclass(frozen=True)
class Criteria7015Thresholds:
    """分布健全性の判定閾値。"""

    accepted_ratio_min: float = 0.1
    accepted_ratio_max: float = 0.9

    saturation_ratio_max: float = 0.2

    discrete_ratio_max: float = 0.5


def evaluate_criteria7015(
    metrics: DistributionHealthMetrics,
    thresholds: Criteria7015Thresholds | None = None,
) -> Criteria7015Result:
    """分布健全性を Criteria7015 に基づいて評価する。

    Args:
        metrics:
            distribution_health で算出されたメトリクス。
        thresholds:
            判定閾値（未指定ならデフォルト）。

    Returns:
        Criteria7015Result:
            validation_pass と失敗理由一覧。
    """

    if thresholds is None:
        thresholds = Criteria7015Thresholds()

    reasons: list[str] = []

    # accepted_ratio チェック
    if metrics.accepted_ratio < thresholds.accepted_ratio_min:
        reasons.append(f"accepted_ratio too low: {metrics.accepted_ratio:.3f}")

    if metrics.accepted_ratio > thresholds.accepted_ratio_max:
        reasons.append(f"accepted_ratio too high: {metrics.accepted_ratio:.3f}")

    # saturation_ratio チェック
    if metrics.saturation_ratio > thresholds.saturation_ratio_max:
        reasons.append(f"saturation_ratio too high: {metrics.saturation_ratio:.3f}")

    # discrete_ratio チェック
    if metrics.discrete_ratio > thresholds.discrete_ratio_max:
        reasons.append(f"discrete_ratio too high: {metrics.discrete_ratio:.3f}")

    validation_pass = len(reasons) == 0

    return Criteria7015Result(
        validation_pass=validation_pass,
        reasons=reasons,
    )
