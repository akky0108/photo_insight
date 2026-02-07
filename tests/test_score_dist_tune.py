# tests/test_score_dist_tune.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- make tools importable ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.score_dist_tune import (  # noqa: E402
    Thresholds,
    score_from_raw,
    resolve_raw_spec,
    infer_direction_by_score,
)


def test_score_from_raw_higher_is_better_basic():
    thr = Thresholds(10.0, 20.0, 30.0, 40.0)
    # higher is better
    assert score_from_raw(0.0, thr, higher_is_better=True) == 0.0
    assert score_from_raw(10.0, thr, higher_is_better=True) == 0.25
    assert score_from_raw(19.999, thr, higher_is_better=True) == 0.25
    assert score_from_raw(20.0, thr, higher_is_better=True) == 0.5
    assert score_from_raw(30.0, thr, higher_is_better=True) == 0.75
    assert score_from_raw(40.0, thr, higher_is_better=True) == 1.0
    assert score_from_raw(999.0, thr, higher_is_better=True) == 1.0


def test_score_from_raw_lower_is_better_inverts_assignment():
    thr = Thresholds(10.0, 20.0, 30.0, 40.0)
    # lower is better (e.g. sigma)
    assert score_from_raw(0.0, thr, higher_is_better=False) == 1.0
    assert score_from_raw(10.0, thr, higher_is_better=False) == 0.75
    assert score_from_raw(20.0, thr, higher_is_better=False) == 0.5
    assert score_from_raw(30.0, thr, higher_is_better=False) == 0.25
    assert score_from_raw(40.0, thr, higher_is_better=False) == 0.0
    assert score_from_raw(999.0, thr, higher_is_better=False) == 0.0


def test_resolve_raw_spec_noise_prefers_noise_raw_over_sigma_used():
    # noise_raw と noise_sigma_used が両方ある場合、noise_raw を選ぶ
    df = pd.DataFrame(
        {
            "noise_raw": [1.0, 2.0, 3.0],
            "noise_sigma_used": [0.1, 0.2, 0.3],
            "noise_score": [0.0, 0.5, 1.0],
        }
    )
    raw_col, raw_source, higher_is_better, note = resolve_raw_spec(df, "noise_score")
    assert raw_col == "noise_raw"
    assert raw_source == "noise_raw"
    assert note == "raw"
    # 相関が正なら True になり得る（ここでは単純に正相関に近いので True 期待）
    assert higher_is_better is True


def test_resolve_raw_spec_noise_uses_sigma_when_noise_raw_missing():
    df = pd.DataFrame(
        {
            "noise_sigma_used": [0.1, 0.2, 0.3],
            "noise_score": [1.0, 0.5, 0.0],  # sigma が大きいほど悪い（負相関）
        }
    )
    raw_col, raw_source, higher_is_better, note = resolve_raw_spec(df, "noise_score")
    assert raw_col == "noise_sigma_used"
    assert raw_source == "noise_sigma_used"
    assert note == "sigma_used"
    assert higher_is_better is False  # sigma は低いほど良い固定


def test_infer_direction_by_score_positive_and_negative_correlation():
    # 30点以上にして、infer_direction_by_score のデフォルト分岐(len<30)を回避する
    n = 60
    raw = pd.Series(np.arange(n, dtype=float))

    # 正相関 → True
    score_pos = pd.Series(np.linspace(0.0, 1.0, n), dtype=float)
    assert infer_direction_by_score(raw, score_pos) is True

    # 負相関 → False
    score_neg = pd.Series(np.linspace(1.0, 0.0, n), dtype=float)
    assert infer_direction_by_score(raw, score_neg) is False


def test_resolve_raw_spec_non_noise_uses_metric_raw_if_present():
    df = pd.DataFrame(
        {
            "contrast_raw": [10, 20, 30],
            "contrast_score": [0.0, 0.5, 1.0],
        }
    )
    raw_col, raw_source, higher_is_better, note = resolve_raw_spec(df, "contrast_score")
    assert raw_col == "contrast_raw"
    assert raw_source == "raw"
    assert higher_is_better is True
    assert note == "raw"


def test_infer_direction_by_score_defaults_true_when_too_few_samples():
    raw = pd.Series([1, 2, 3, 4, 5], dtype=float)
    score_neg = pd.Series([1.0, 0.75, 0.5, 0.25, 0.0], dtype=float)
    assert infer_direction_by_score(raw, score_neg) is True
