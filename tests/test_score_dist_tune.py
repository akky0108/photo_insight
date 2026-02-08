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
    build_raw_transform_spec,
    infer_direction_by_score,
    resolve_raw_col,
    score_from_raw,
    build_evaluator_config_from_chosen_params,
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


def test_resolve_raw_col_noise_prefers_sigma_used_over_noise_raw():
    # noise_raw と noise_sigma_used が両方ある場合、sigma_used を選ぶ（axis=sigma）
    df = pd.DataFrame(
        {
            "noise_raw": [1.0, 2.0, 3.0],
            "noise_sigma_used": [0.1, 0.2, 0.3],
            "noise_score": [0.0, 0.5, 1.0],
        }
    )
    raw_col, raw_source, raw_direction, raw_transform, meta = resolve_raw_col(df, "noise_score")
    assert raw_col == "noise_sigma_used"
    assert raw_source == "noise_sigma_used"
    assert raw_direction == "lower_is_better"
    assert raw_transform == "lower_is_better"
    assert meta.get("direction_note") == "sigma_axis"
    assert isinstance(meta, dict)


def test_resolve_raw_col_noise_uses_sigma_when_noise_raw_missing():
    df = pd.DataFrame(
        {
            "noise_sigma_used": [0.1, 0.2, 0.3],
            "noise_score": [1.0, 0.5, 0.0],
        }
    )
    raw_col, raw_source, raw_direction, raw_transform, meta = resolve_raw_col(df, "noise_score")
    assert raw_col == "noise_sigma_used"
    assert raw_source == "noise_sigma_used"
    assert raw_direction == "lower_is_better"
    assert raw_transform == "lower_is_better"
    assert meta.get("direction_note") == "sigma_axis"


def test_infer_direction_by_score_positive_and_negative_correlation():
    # 30点以上にして、infer_direction_by_score のデフォルト分岐(n<30)を回避する
    n = 60
    raw = pd.Series(np.arange(n, dtype=float))

    # 正相関 → True
    score_pos = pd.Series(np.linspace(0.0, 1.0, n), dtype=float)
    higher, corr, nn, inferred = infer_direction_by_score(raw, score_pos)
    assert higher is True
    assert inferred is True
    assert nn == n
    assert corr is not None
    assert corr > 0

    # 負相関 → False
    score_neg = pd.Series(np.linspace(1.0, 0.0, n), dtype=float)
    higher, corr, nn, inferred = infer_direction_by_score(raw, score_neg)
    assert higher is False
    assert inferred is True
    assert nn == n
    assert corr is not None
    assert corr < 0


def test_resolve_raw_col_non_noise_uses_metric_raw_if_present():
    df = pd.DataFrame(
        {
            "contrast_raw": [10, 20, 30],
            "contrast_score": [0.0, 0.5, 1.0],
        }
    )
    raw_col, raw_source, raw_direction, raw_transform, meta = resolve_raw_col(df, "contrast_score")
    assert raw_col == "contrast_raw"
    assert raw_source == "raw"
    assert raw_direction == "higher_is_better"
    assert raw_transform == "identity"
    assert isinstance(meta, dict)


def test_infer_direction_by_score_defaults_true_when_too_few_samples():
    raw = pd.Series([1, 2, 3, 4, 5], dtype=float)
    score_neg = pd.Series([1.0, 0.75, 0.5, 0.25, 0.0], dtype=float)
    higher, corr, n, inferred = infer_direction_by_score(raw, score_neg)
    assert higher is True
    assert inferred is False
    assert n == 5
    assert corr is None


def test_raw_spec_contains_direction_and_transform():
    rs = build_raw_transform_spec(
        "noise",
        "noise_sigma_used",
        higher_is_better=False,
        raw_source="noise_sigma_used",
        direction_meta={"direction_inferred": False},
    )
    assert rs["raw_direction"] == "lower_is_better"
    assert rs["raw_transform"] == "lower_is_better"


def test_build_evaluator_config_emits_noise_suggestions_only():
    chosen = {
        "noise": {
            "thresholds": [0.1, 0.2, 0.3, 0.4],
            "source": "auto",
            "score_col": "noise_score",
            "raw_col": "noise_sigma_used",
            "raw_source": "noise_sigma_used",
            "higher_is_better": False,
            "raw_spec": {
                "raw_direction": "lower_is_better",
                "raw_transform": "lower_is_better",
                "source_raw_col": "noise_sigma_used",
                "effective_raw_col": "noise_sigma_used",
                "raw_source": "noise_sigma_used",
                "higher_is_better": False,
                "direction_inferred": False,
                "corr": None,
                "n_for_corr": 0,
                "direction_note": "sigma_axis",
            },
        },
        "contrast": {
            "thresholds": [1, 2, 3, 4],
            "source": "auto",
            "score_col": "contrast_score",
            "raw_col": "contrast_raw",
            "raw_source": "raw",
            "higher_is_better": True,
            "raw_spec": {"raw_direction": "higher_is_better", "raw_transform": "identity"},
        },
    }

    cfg = build_evaluator_config_from_chosen_params(chosen)

    # noise は SSOT に出さない
    assert "noise" not in cfg
    assert "noise_suggestions" in cfg

    ns = cfg["noise_suggestions"]["noise"]
    assert ns["axis"] == "sigma"
    assert ns["thresholds_5bin"] == [0.1, 0.2, 0.3, 0.4]
    assert ns["good_sigma_suggestion"] == 0.2
    assert ns["warn_sigma_suggestion"] == 0.4
