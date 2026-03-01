from photo_insight.cli.run_batch import _apply_runtime_overrides


class DummyProc:
    pass


def test_apply_runtime_overrides_injects_target_date():
    proc = DummyProc()
    injected = {"date": "2026-02-17"}
    _apply_runtime_overrides(proc, injected)
    assert getattr(proc, "date") == "2026-02-17"
    assert getattr(proc, "target_date") == "2026-02-17"
