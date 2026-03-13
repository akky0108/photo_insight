from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from photo_insight.cli import run_batch


def test_resolve_session_name_from_injected_prefers_target_dir() -> None:
    session = run_batch._resolve_session_name_from_injected(
        {
            "date": "2026-02-17",
            "target_dir": "/work/input/2026-02-17",
        }
    )
    assert session == "2026-02-17"


def test_resolve_session_name_from_injected_falls_back_to_date() -> None:
    session = run_batch._resolve_session_name_from_injected(
        {
            "date": "2026-02-17",
        }
    )
    assert session == "2026-02-17"


def test_resolve_session_name_from_injected_defaults_to_all() -> None:
    session = run_batch._resolve_session_name_from_injected({})
    assert session == "ALL"


def test_infer_nef_output_csv_path_prefers_same_run_artifact(
    tmp_path: Path,
) -> None:
    session = "2026-02-17"
    csv_path = (
        tmp_path
        / "run-out"
        / "artifacts"
        / "nef"
        / session
        / f"{session}_raw_exif_data.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("dummy", encoding="utf-8")

    class DummyRunCtx:
        out_dir = str(tmp_path / "run-out")

    class DummyProc:
        run_ctx = DummyRunCtx()
        project_root = str(tmp_path)

    result = run_batch._infer_nef_output_csv_path(
        proc=DummyProc(),
        injected={"date": session},
    )

    assert result == str(csv_path)


def test_infer_nef_output_csv_path_falls_back_to_runs_latest(
    tmp_path: Path,
) -> None:
    session = "2026-02-17"
    csv_path = (
        tmp_path
        / "runs"
        / "latest"
        / "nef"
        / session
        / f"{session}_raw_exif_data.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("dummy", encoding="utf-8")

    class DummyProc:
        run_ctx = None
        project_root = str(tmp_path)

    result = run_batch._infer_nef_output_csv_path(
        proc=DummyProc(),
        injected={"date": session},
    )

    assert result == str(csv_path)


def test_build_stage_result_for_nef_includes_output_csv_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv"

    def fake_infer_nef_output_csv_path(proc: Any, injected: dict[str, Any]) -> str:
        return expected

    monkeypatch.setattr(
        run_batch,
        "_infer_nef_output_csv_path",
        fake_infer_nef_output_csv_path,
    )

    class DummyProc:
        pass

    result = run_batch._build_stage_result(
        processor_spec="nef",
        proc=DummyProc(),
        injected={"date": "2026-02-17"},
    )

    assert result == {
        "name": "nef",
        "status": "success",
        "output_csv_path": expected,
    }


def test_build_stage_result_for_portrait_quality_has_minimum_fields() -> None:
    class DummyProc:
        pass

    result = run_batch._build_stage_result(
        processor_spec="portrait_quality",
        proc=DummyProc(),
        injected={"date": "2026-02-17"},
    )

    assert result == {
        "name": "portrait_quality",
        "status": "success",
    }


def test_run_pipeline_chain_passes_nef_csv_to_portrait_quality(
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

    assert calls[0]["exec_kwargs"] == {
        "append_mode": True,
    }
    assert calls[1]["exec_kwargs"] == {
        "append_mode": True,
        "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
    }


def test_run_pipeline_chain_raises_when_nef_output_csv_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: dict[str, Any],
        exec_kwargs: dict[str, Any],
        injected: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(processor_spec)

        if processor_spec == "nef":
            return {
                "name": "nef",
                "status": "success",
                "output_csv_path": None,
            }

        return {
            "name": "portrait_quality",
            "status": "success",
        }

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    with pytest.raises(FileNotFoundError, match="NEF output CSV path could not be resolved"):
        run_batch.run_pipeline_chain(
            stages=["nef", "portrait_quality"],
            ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
            exec_kwargs={"append_mode": True},
            injected={"date": "2026-02-17"},
        )

    assert calls == ["nef"]


def test_run_pipeline_chain_requires_previous_nef_stage_result() -> None:
    with pytest.raises(RuntimeError, match="requires a previous nef stage result"):
        run_batch.run_pipeline_chain(
            stages=["portrait_quality"],
            ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
            exec_kwargs={"append_mode": True},
            injected={"date": "2026-02-17"},
        )