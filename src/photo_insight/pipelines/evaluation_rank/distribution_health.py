from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DistributionHealthMetrics:
    """評価結果CSVから算出した分布健全性メトリクス。"""

    total_count: int
    accepted_count: int
    accepted_ratio: float
    saturation_ratio: float
    discrete_ratio: float


def calculate_distribution_health(
    csv_path: str | Path,
    *,
    score_column: str = "score",
    decision_column: str = "decision",
    accepted_value: str = "accepted",
    saturation_min_score: float = 0.0,
    saturation_max_score: float = 100.0,
) -> DistributionHealthMetrics:
    """評価結果CSVから分布健全性メトリクスを算出する。

    Args:
        csv_path:
            evaluation_results_*.csv のパス。
        score_column:
            スコア列名。
        decision_column:
            判定列名。
        accepted_value:
            accepted とみなす値。
        saturation_min_score:
            下限飽和とみなすスコア。
        saturation_max_score:
            上限飽和とみなすスコア。

    Returns:
        DistributionHealthMetrics:
            accepted_ratio / saturation_ratio / discrete_ratio を含む結果。

    Raises:
        FileNotFoundError:
            CSV が存在しない場合。
        ValueError:
            必須列が不足している場合。
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    required_columns = {score_column, decision_column}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing_columns)))

    total_count = len(df)

    if total_count == 0:
        return DistributionHealthMetrics(
            total_count=0,
            accepted_count=0,
            accepted_ratio=0.0,
            saturation_ratio=0.0,
            discrete_ratio=0.0,
        )

    scores = pd.to_numeric(df[score_column], errors="coerce")

    accepted_count = int((df[decision_column] == accepted_value).sum())
    accepted_ratio = accepted_count / total_count

    saturation_count = int(((scores <= saturation_min_score) | (scores >= saturation_max_score)).sum())
    saturation_ratio = saturation_count / total_count

    score_counts = scores.value_counts(dropna=True)

    if score_counts.empty:
        discrete_ratio = 0.0
    else:
        discrete_ratio = int(score_counts.max()) / total_count

    return DistributionHealthMetrics(
        total_count=total_count,
        accepted_count=accepted_count,
        accepted_ratio=accepted_ratio,
        saturation_ratio=saturation_ratio,
        discrete_ratio=discrete_ratio,
    )
