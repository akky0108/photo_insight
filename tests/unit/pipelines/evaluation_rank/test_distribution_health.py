import pandas as pd
from pathlib import Path

from photo_insight.pipelines.evaluation_rank.distribution_health import (
    calculate_distribution_health,
)


def test_distribution_health_basic(tmp_path: Path):
    csv_path = tmp_path / "test.csv"

    df = pd.DataFrame(
        {
            "score": [10, 50, 90, 100],
            "decision": ["accepted", "rejected", "accepted", "accepted"],
        }
    )
    df.to_csv(csv_path, index=False)

    result = calculate_distribution_health(csv_path)

    assert result.total_count == 4
    assert result.accepted_count == 3
    assert result.accepted_ratio == 3 / 4


def test_distribution_health_empty(tmp_path: Path):
    csv_path = tmp_path / "test.csv"

    df = pd.DataFrame(columns=["score", "decision"])
    df.to_csv(csv_path, index=False)

    result = calculate_distribution_health(csv_path)

    assert result.total_count == 0
    assert result.accepted_ratio == 0.0


def test_distribution_health_missing_column(tmp_path: Path):
    csv_path = tmp_path / "test.csv"

    df = pd.DataFrame({"score": [10, 20]})
    df.to_csv(csv_path, index=False)

    import pytest

    with pytest.raises(ValueError):
        calculate_distribution_health(csv_path)
