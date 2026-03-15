from __future__ import annotations

import pytest

from photo_insight.cli.run_batch import _extract_runtime_overrides


def test_extract_runtime_overrides_extracts_date() -> None:
    exec_kwargs = {"date": "2026-02-17"}

    injected = _extract_runtime_overrides(exec_kwargs)

    assert injected == {"date": "2026-02-17"}
    assert exec_kwargs == {}


def test_extract_runtime_overrides_extracts_run_date_as_date() -> None:
    exec_kwargs = {"run_date": "2026-02-18"}

    injected = _extract_runtime_overrides(exec_kwargs)

    assert injected == {"date": "2026-02-18"}
    assert exec_kwargs == {}


def test_extract_runtime_overrides_extracts_target_dir() -> None:
    exec_kwargs = {"target_dir": "/work/input/2026-02-17"}

    injected = _extract_runtime_overrides(exec_kwargs)

    assert injected == {"target_dir": "/work/input/2026-02-17"}
    assert exec_kwargs == {}


def test_extract_runtime_overrides_extracts_dir_alias_as_target_dir() -> None:
    exec_kwargs = {"dir": "/work/input/2026-02-17"}

    injected = _extract_runtime_overrides(exec_kwargs)

    assert injected == {"target_dir": "/work/input/2026-02-17"}
    assert exec_kwargs == {}


def test_extract_runtime_overrides_extracts_nef_prefixed_options() -> None:
    exec_kwargs = {
        "nef_incremental": True,
        "nef_limit": 10,
        "max_images": 5,
    }

    injected = _extract_runtime_overrides(exec_kwargs)

    assert injected == {
        "nef_incremental": True,
        "nef_limit": 10,
    }
    assert exec_kwargs == {"max_images": 5}


def test_extract_runtime_overrides_extracts_all_supported_runtime_values() -> None:
    exec_kwargs = {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026-02-17",
        "nef_incremental": True,
        "nef_max_files": 20,
        "max_images": 10,
    }

    injected = _extract_runtime_overrides(exec_kwargs)

    assert injected == {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026-02-17",
        "nef_incremental": True,
        "nef_max_files": 20,
    }
    assert exec_kwargs == {"max_images": 10}


def test_extract_runtime_overrides_rejects_invalid_date_format() -> None:
    exec_kwargs = {"date": "20260217"}

    with pytest.raises(ValueError, match="Invalid date format"):
        _extract_runtime_overrides(exec_kwargs)
