from __future__ import annotations

from photo_insight.cli.run_batch import _apply_runtime_overrides


class DummyProc:
    """BaseBatchProcessor を使わずに override 注入をテストするためのダミー"""

    pass


def test_apply_runtime_overrides_injects_target_date() -> None:
    proc = DummyProc()
    injected = {"date": "2026-02-17"}

    _apply_runtime_overrides(proc, injected)

    assert getattr(proc, "date") == "2026-02-17"
    assert getattr(proc, "target_date") == "2026-02-17"


def test_apply_runtime_overrides_injects_target_dir() -> None:
    proc = DummyProc()
    injected = {"target_dir": "/work/input/2026-02-17"}

    _apply_runtime_overrides(proc, injected)

    assert str(getattr(proc, "target_dir")) == "/work/input/2026-02-17"


def test_apply_runtime_overrides_injects_nef_prefixed_options() -> None:
    proc = DummyProc()
    injected = {
        "nef_incremental": True,
        "nef_limit": 10,
    }

    _apply_runtime_overrides(proc, injected)

    assert getattr(proc, "nef_incremental") is True
    assert getattr(proc, "nef_limit") == 10
