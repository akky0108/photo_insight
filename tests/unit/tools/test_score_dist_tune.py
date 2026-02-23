# tests/test_score_dist_tune.py
from __future__ import annotations
import pytest
pytestmark = pytest.mark.heavy

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
    validate_dataframe_contract,
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
            "raw_spec": {
                "raw_direction": "higher_is_better",
                "raw_transform": "identity",
            },
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


def test_build_evaluator_config_emits_local_contrast_thresholds():
    chosen = {
        "local_contrast": {
            "thresholds": [0.01, 0.02, 0.03, 0.04],
            "source": "auto",
            "score_col": "local_contrast_score",
            "raw_col": "local_contrast_raw",
            "raw_source": "raw",
            "higher_is_better": True,
            "raw_spec": {
                "raw_direction": "higher_is_better",
                "raw_transform": "identity",
            },
        }
    }

    cfg = build_evaluator_config_from_chosen_params(chosen)

    assert "local_contrast" in cfg
    assert "discretize_thresholds_raw" in cfg["local_contrast"]

    m = cfg["local_contrast"]["discretize_thresholds_raw"]
    assert set(m.keys()) == {"poor", "fair", "good", "excellent"}
    assert m["poor"] == 0.01
    assert m["fair"] == 0.02
    assert m["good"] == 0.03
    assert m["excellent"] == 0.04


def test_build_raw_spec_direction_meta_is_updatable():
    # direction_meta を後段で update して raw_spec に反映する運用を前提にする
    rs = build_raw_transform_spec(
        "blurriness",
        "blurriness_raw",
        higher_is_better=True,
        raw_source="raw",
        direction_meta={
            "direction_inferred": True,
            "corr": -0.5,
            "n_for_corr": 60,
            "direction_note": "corr_check_mismatch",
        },
    )
    assert rs["direction_inferred"] is True
    assert rs["corr"] == -0.5
    assert rs["n_for_corr"] == 60
    assert rs["direction_note"] == "corr_check_mismatch"


def test_validate_dataframe_contract_blurriness_missing_contract_cols_warns_but_ok(
    capsys,
):
    # 欠損は “今は” WARN で許容（段階導入）
    df = pd.DataFrame(
        {
            "sharpness_score": [0.0, 0.5, 1.0],
            "sharpness_raw": [1.0, 2.0, 3.0],
            "blurriness_score": [0.0, 0.5, 1.0],
            "blurriness_raw": [0.1, 0.2, 0.3],
            "contrast_score": [0.0, 0.5, 1.0],
            "contrast_raw": [10, 20, 30],
            "noise_score": [1.0, 0.5, 0.0],
            "noise_sigma_used": [0.1, 0.2, 0.3],
            # ★追加（TARGET_SCORE_COLS に含まれているため）
            "local_contrast_score": [0.0, 0.5, 1.0],
            "local_contrast_raw": [0.01, 0.02, 0.03],
            "face_sharpness_score": [0.0, 0.5, 1.0],
            "face_sharpness_raw": [1.0, 2.0, 3.0],
            "face_blurriness_score": [0.0, 0.5, 1.0],
            "face_blurriness_raw": [0.1, 0.2, 0.3],
            "face_contrast_score": [0.0, 0.5, 1.0],
            "face_contrast_raw": [10, 20, 30],
            # blurriness contract cols are intentionally missing
        }
    )
    assert validate_dataframe_contract(df) is True
    err = capsys.readouterr().err
    assert "missing blurriness contract columns" in err


def test_validate_dataframe_contract_blurriness_contract_value_mismatch_fails(capsys):
    # blurriness contract が値不一致なら FAIL（止める）
    df = pd.DataFrame(
        {
            "sharpness_score": [0.0, 0.5, 1.0],
            "sharpness_raw": [1.0, 2.0, 3.0],
            "blurriness_score": [0.0, 0.5, 1.0],
            "blurriness_raw": [0.1, 0.2, 0.3],
            "contrast_score": [0.0, 0.5, 1.0],
            "contrast_raw": [10, 20, 30],
            "noise_score": [1.0, 0.5, 0.0],
            "noise_sigma_used": [0.1, 0.2, 0.3],
            # ★追加（TARGET_SCORE_COLS に含まれているため）
            "local_contrast_score": [0.0, 0.5, 1.0],
            "local_contrast_raw": [0.01, 0.02, 0.03],
            "face_sharpness_score": [0.0, 0.5, 1.0],
            "face_sharpness_raw": [1.0, 2.0, 3.0],
            "face_blurriness_score": [0.0, 0.5, 1.0],
            "face_blurriness_raw": [0.1, 0.2, 0.3],
            "face_contrast_score": [0.0, 0.5, 1.0],
            "face_contrast_raw": [10, 20, 30],
            # contract cols present but wrong
            "blurriness_raw_direction": ["lower_is_better"] * 3,  # ← NG
            "blurriness_raw_transform": ["identity"] * 3,
            "blurriness_higher_is_better": [True] * 3,
        }
    )
    assert validate_dataframe_contract(df) is False
    err = capsys.readouterr().err
    assert "Contract violation" in err
    assert "blurriness_raw_direction" in err
