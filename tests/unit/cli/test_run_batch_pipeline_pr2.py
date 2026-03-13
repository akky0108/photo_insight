from __future__ import annotations

from typing import Any

import pytest

from photo_insight.cli import run_batch


def test_run_single_processor_resolves_and_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    class DummyProcessor:
        def __init__(self, **kwargs: Any) -> None:
            called["ctor_kwargs"] = kwargs

        def execute(self, **kwargs: Any) -> None:
            called["exec_kwargs"] = kwargs

    def fake_resolve_processor(spec: str) -> type[DummyProcessor]:
        called["processor_spec"] = spec
        return DummyProcessor

    def fake_apply_runtime_overrides(proc: Any, injected: dict[str, Any]) -> None:
        called["injected"] = injected
        called["proc"] = proc

    def fake_build_stage_result(
        processor_spec: str,
        proc: Any,
        injected: dict[str, Any],
    ) -> dict[str, Any]:
        called["build_stage_result_args"] = {
            "processor_spec": processor_spec,
            "proc": proc,
            "injected": injected,
        }
        return {
            "name": "nef",
            "status": "success",
            "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
        }

    monkeypatch.setattr(run_batch, "resolve_processor", fake_resolve_processor)
    monkeypatch.setattr(run_batch, "_apply_runtime_overrides", fake_apply_runtime_overrides)
    monkeypatch.setattr(run_batch, "_build_stage_result", fake_build_stage_result)

    result = run_batch.run_single_processor(
        processor_spec="nef",
        ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
        exec_kwargs={"append_mode": True},
        injected={"date": "2026-02-17"},
    )

    assert result == {
        "name": "nef",
        "status": "success",
        "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
    }
    assert called["processor_spec"] == "nef"
    assert called["ctor_kwargs"] == {
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
    }
    assert called["exec_kwargs"] == {"append_mode": True}
    assert called["injected"] == {"date": "2026-02-17"}
    assert called["build_stage_result_args"]["processor_spec"] == "nef"


def test_run_pipeline_chain_executes_stages_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: dict[str, Any],
        exec_kwargs: dict[str, Any],
        injected: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(
            {
                "processor_spec": processor_spec,
                "ctor_kwargs": ctor_kwargs,
                "exec_kwargs": exec_kwargs,
                "injected": injected,
            }
        )

        if processor_spec == "nef":
            return {
                "name": "nef",
                "status": "success",
                "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
            }

        return {
            "name": "portrait_quality",
            "status": "success",
        }

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    rc = run_batch.run_pipeline_chain(
        stages=["nef", "portrait_quality"],
        ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
        exec_kwargs={"append_mode": True},
        injected={"date": "2026-02-17"},
    )

    assert rc == 0
    assert [x["processor_spec"] for x in calls] == ["nef", "portrait_quality"]

    assert calls[0]["ctor_kwargs"] == {
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
    }
    assert calls[0]["exec_kwargs"] == {"append_mode": True}
    assert calls[0]["injected"] == {"date": "2026-02-17"}

    assert calls[1]["ctor_kwargs"] == {
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
    }
    assert calls[1]["exec_kwargs"] == {
        "append_mode": True,
        "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
    }
    assert calls[1]["injected"] == {"date": "2026-02-17"}


def test_run_pipeline_chain_stops_when_first_stage_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: dict[str, Any],
        exec_kwargs: dict[str, Any],
        injected: dict[str, Any],
    ) -> int:
        calls.append(processor_spec)
        if processor_spec == "nef":
            raise RuntimeError("nef failed")
        return 0

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    with pytest.raises(RuntimeError, match="nef failed"):
        run_batch.run_pipeline_chain(
            stages=["nef", "portrait_quality"],
            ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
            exec_kwargs={"append_mode": True},
            injected={"date": "2026-02-17"},
        )

    assert calls == ["nef"]


def test_main_pipeline_execution_calls_run_pipeline_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_run_pipeline_chain(
        stages: list[str],
        ctor_kwargs: dict[str, Any],
        exec_kwargs: dict[str, Any],
        injected: dict[str, Any],
    ) -> int:
        called["stages"] = stages
        called["ctor_kwargs"] = ctor_kwargs
        called["exec_kwargs"] = exec_kwargs
        called["injected"] = injected
        return 0

    monkeypatch.setattr(run_batch, "run_pipeline_chain", fake_run_pipeline_chain)

    rc = run_batch.main(
        [
            "--pipeline",
            "nef,portrait_quality",
            "--config",
            "config/config.prod.yaml",
            "--max-workers",
            "3",
            "--date",
            "2026-02-17",
            "--target-dir",
            "/work/input/2026-02-17",
            "--append-mode",
            "true",
        ]
    )

    assert rc == 0
    assert called["stages"] == ["nef", "portrait_quality"]
    assert called["ctor_kwargs"] == {
        "config_path": "config/config.prod.yaml",
        "max_workers": 3,
    }
    assert called["exec_kwargs"] == {
        "append_mode": True,
    }
    assert called["injected"] == {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026-02-17",
    }


def test_main_pipeline_dry_run_still_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = run_batch.main(
        [
            "--pipeline",
            "nef,portrait_quality",
            "--dry-run",
            "--date",
            "2026-02-17",
            "--target-dir",
            "/work/input/2026-02-17",
        ]
    )

    assert rc == 0

    out = capsys.readouterr().out
    assert "[dry-run] pipeline = ['nef', 'portrait_quality']" in out
    assert "[dry-run] exec_kwargs = {}" in out
    assert "'date': '2026-02-17'" in out
    assert "'target_dir': '/work/input/2026-02-17'" in out


def test_main_single_processor_execution_calls_run_single_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, Any] = {}

    def fake_resolve_processor(spec: str) -> Any:
        class DummyProcessor:
            pass

        return DummyProcessor

    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: dict[str, Any],
        exec_kwargs: dict[str, Any],
        injected: dict[str, Any],
    ) -> int:
        called["processor_spec"] = processor_spec
        called["ctor_kwargs"] = ctor_kwargs
        called["exec_kwargs"] = exec_kwargs
        called["injected"] = injected
        return 0

    monkeypatch.setattr(run_batch, "resolve_processor", fake_resolve_processor)
    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    rc = run_batch.main(
        [
            "--processor",
            "nef",
            "--config",
            "config/config.prod.yaml",
            "--max-workers",
            "4",
            "--date",
            "2026-02-17",
            "--target-dir",
            "/work/input/2026-02-17",
            "--append-mode",
            "true",
        ]
    )

    assert rc == 0
    assert called["processor_spec"] == "nef"
    assert called["ctor_kwargs"] == {
        "config_path": "config/config.prod.yaml",
        "max_workers": 4,
    }
    assert called["exec_kwargs"] == {
        "append_mode": True,
    }
    assert called["injected"] == {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026-02-17",
    }
