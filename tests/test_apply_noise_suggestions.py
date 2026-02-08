from __future__ import annotations

from pathlib import Path
import json

from tools.apply_noise_suggestions import (
    extract_noise_sigmas_from_params,
    patch_yaml_text_config,
)


def test_extract_noise_sigmas_from_params_uses_thresholds_index_1_and_3():
    data = {
        "noise": {
            "thresholds": [0.001, 0.002, 0.003, 0.004],
            "source": "auto",
            "raw_spec": {"raw_direction": "lower_is_better", "raw_transform": "lower_is_better"},
        }
    }
    sig = extract_noise_sigmas_from_params(data)
    assert sig.good_sigma == 0.002
    assert sig.warn_sigma == 0.004
    assert sig.raw_direction == "lower_is_better"
    assert sig.raw_transform == "lower_is_better"


def test_patch_yaml_text_config_adds_noise_block_if_missing():
    old = "sharpness:\n  discretize_thresholds_raw:\n    poor: 1\n"
    sig = extract_noise_sigmas_from_params(
        {"noise": {"thresholds": [1.0, 2.0, 3.0, 4.0], "source": "auto", "raw_spec": {}}}
    )
    new = patch_yaml_text_config(old, sig)
    assert "noise:\n" in new
    assert "  good_sigma: 2" in new
    assert "  warn_sigma: 4" in new


def test_patch_yaml_text_config_updates_existing_noise_block_values():
    old = (
        "noise:\n"
        "  good_sigma: 0.01\n"
        "  warn_sigma: 0.02\n"
        "contrast:\n"
        "  discretize_thresholds_raw:\n"
        "    poor: 1\n"
    )
    sig = extract_noise_sigmas_from_params(
        {"noise": {"thresholds": [0.1, 0.111, 0.2, 0.222], "source": "auto", "raw_spec": {}}}
    )
    new = patch_yaml_text_config(old, sig)
    assert "  good_sigma: 0.111" in new
    assert "  warn_sigma: 0.222" in new
    # contrast block remains
    assert "contrast:\n" in new


def test_patch_yaml_text_config_inserts_missing_keys_inside_noise_block():
    old = (
        "noise:\n"
        "  # comment\n"
        "contrast:\n"
        "  discretize_thresholds_raw:\n"
        "    poor: 1\n"
    )
    sig = extract_noise_sigmas_from_params(
        {"noise": {"thresholds": [0.1, 0.2, 0.3, 0.4], "source": "auto", "raw_spec": {}}}
    )
    new = patch_yaml_text_config(old, sig)
    assert "noise:\n" in new
    assert "  good_sigma: 0.2" in new
    assert "  warn_sigma: 0.4" in new
