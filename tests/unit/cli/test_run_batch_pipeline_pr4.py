from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from photo_insight.cli import run_batch


class DummyProcessor:
    runtime_param_names = ()

    def __init__(self, **kwargs: Any) -> None:
        self.ctor_kwargs = kwargs
        self.execute_calls: List[Dict[str, Any]] = []
        self.project_root = Path("/tmp/project")
        self.run_ctx = None

    def execute(self, **kwargs: Any) -> None:
        self.execute_calls.append(dict(kwargs))


class DummyNEFProcessor(DummyProcessor):
    runtime_param_names = ("target_date", "target_dir", "nef_incremental", "nef_max_files", "nef_dry_run")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.processed_count = 5
        self.run_ctx = type("RunCtx", (), {"out_dir": "/tmp/project/runs/latest"})()


@pytest.fixture
def ctor_kwargs() -> Dict[str, Any]:
    return {
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
    }


@pytest.fixture
def runtime_overrides() -> Dict[str, Any]:
    return {
        "date": "2026-02-17",
    }


def test_build_stage_result_for_nef_returns_extended_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_csv = "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv"

    monkeypatch.setattr(
        run_batch,
        "_infer_nef_output_csv_path",
        lambda proc, runtime_overrides: expected_csv,
    )

    class DummyNEFProc:
        processed_count = 5
        project_root = "/tmp/project"

    proc = DummyNEFProc()

    result = run_batch._build_stage_result(
        processor_spec="nef",
        processor_cls=DummyNEFProcessor,
        proc=proc,
        runtime_overrides={"date": "2026-02-17"},
        exec_kwargs={"max_images": 5},
    )

    assert result["name"] == "nef"
    assert result["status"] == "success"
    assert result["input_csv_path"] is None
    assert result["output_csv_path"] == expected_csv
    assert result["processed_count"] == 5
    assert result["applied_max_images"] == 5
    assert result["message"] is None
    assert result["run_output_dir"] == "/tmp/project/runs/latest"


def test_build_stage_result_for_portrait_quality_returns_extended_fields() -> None:
    class DummyPortraitProc:
        processed_count = 5
        project_root = "/tmp/project"

    class DummyPortraitProcessor:
        __name__ = "PortraitQualityBatchProcessor"
        runtime_param_names = ("date", "target_dir", "max_images", "input_csv_path")

    result = run_batch._build_stage_result(
        processor_spec="portrait_quality",
        processor_cls=DummyPortraitProcessor,
        proc=DummyPortraitProc(),
        runtime_overrides={"date": "2026-02-17"},
        exec_kwargs={
            "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
            "max_images": 5,
        },
    )

    assert result["name"] == "portrait_quality"
    assert result["status"] == "success"
    assert result["input_csv_path"] == "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv"
    assert result["output_csv_path"] is None
    assert result["processed_count"] == 5
    assert result["applied_max_images"] == 5
    assert result["message"] is None
    assert result["run_output_dir"] == "/tmp/project/runs/latest"


def test_run_pipeline_chain_returns_summary_and_passes_max_images_to_portrait_quality_stage(
    monkeypatch: pytest.MonkeyPatch,
    ctor_kwargs: Dict[str, Any],
    runtime_overrides: Dict[str, Any],
) -> None:
    calls: List[Dict[str, Any]] = []

    class FakeNefProcessor:
        runtime_param_names = ("target_date", "target_dir", "nef_incremental", "nef_max_files", "nef_dry_run")

    class FakePortraitProcessor:
        runtime_param_names = ("date", "target_dir", "max_images", "input_csv_path")

    def fake_resolve_processor(spec: str):
        if spec == "nef":
            return FakeNefProcessor
        if spec == "portrait_quality":
            return FakePortraitProcessor
        raise AssertionError(f"unexpected processor: {spec}")

    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: Dict[str, Any],
        exec_kwargs: Dict[str, Any],
        runtime_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        calls.append(
            {
                "processor_spec": processor_spec,
                "ctor_kwargs": dict(ctor_kwargs),
                "exec_kwargs": dict(exec_kwargs),
                "runtime_overrides": dict(runtime_overrides),
            }
        )

        if processor_spec == "nef":
            return {
                "name": "nef",
                "status": "success",
                "input_csv_path": None,
                "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                "processed_count": 5,
                "applied_max_images": exec_kwargs.get("max_images"),
                "message": None,
                "run_output_dir": "/tmp/project/runs/latest",
            }

        if processor_spec == "portrait_quality":
            return {
                "name": "portrait_quality",
                "status": "success",
                "input_csv_path": exec_kwargs.get("input_csv_path"),
                "output_csv_path": None,
                "processed_count": 5,
                "applied_max_images": exec_kwargs.get("max_images"),
                "message": None,
                "run_output_dir": "/tmp/project/runs/latest",
            }

        raise AssertionError(f"unexpected processor: {processor_spec}")

    monkeypatch.setattr(run_batch, "resolve_processor", fake_resolve_processor)
    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    summary = run_batch.run_pipeline_chain(
        stages=["nef", "portrait_quality"],
        ctor_kwargs=ctor_kwargs,
        exec_kwargs={"append_mode": True, "max_images": 5},
        runtime_overrides=runtime_overrides,
    )

    assert summary["pipeline"] == ["nef", "portrait_quality"]
    assert summary["status"] == "success"
    assert len(summary["stages"]) == 2
    assert "duration_sec" in summary
    assert isinstance(summary["duration_sec"], float)
    assert summary["duration_sec"] >= 0.0

    assert [x["processor_spec"] for x in calls] == ["nef", "portrait_quality"]

    assert calls[0]["exec_kwargs"] == {
        "target_date": "2026-02-17",
        "max_images": 5,
    }
    assert calls[1]["exec_kwargs"] == {
        "date": "2026-02-17",
        "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
        "max_images": 5,
    }

    assert summary["stages"][0]["name"] == "nef"
    assert summary["stages"][0]["applied_max_images"] == 5

    assert summary["stages"][1]["name"] == "portrait_quality"
    assert summary["stages"][1]["input_csv_path"] == "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv"
    assert summary["stages"][1]["applied_max_images"] == 5


def test_print_pipeline_summary_outputs_expected_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = {
        "pipeline": ["nef", "portrait_quality"],
        "status": "success",
        "duration_sec": 0.123,
        "stages": [
            {
                "name": "nef",
                "status": "success",
                "input_csv_path": None,
                "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                "processed_count": 5,
                "applied_max_images": 5,
                "message": None,
            },
            {
                "name": "portrait_quality",
                "status": "success",
                "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                "output_csv_path": None,
                "processed_count": 5,
                "applied_max_images": 5,
                "message": None,
            },
        ],
    }

    run_batch._print_pipeline_summary(summary)
    captured = capsys.readouterr()

    assert "[pipeline summary]" in captured.out
    assert "pipeline = nef,portrait_quality" in captured.out
    assert "status = success" in captured.out
    assert "duration_sec = 0.123" in captured.out
    assert "[stage] nef" in captured.out
    assert "applied_max_images = 5" in captured.out
    assert "processed_count = 5" in captured.out
    assert "[stage] portrait_quality" in captured.out
    assert "input_csv_path = /work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv" in captured.out


def test_main_pipeline_prints_summary_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: List[Dict[str, Any]] = []

    monkeypatch.setattr(
        run_batch,
        "run_pipeline_chain",
        lambda stages, ctor_kwargs, exec_kwargs, runtime_overrides: {
            "pipeline": stages,
            "status": "success",
            "duration_sec": 0.123,
            "stages": [
                {
                    "name": "nef",
                    "status": "success",
                    "input_csv_path": None,
                    "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                    "processed_count": 5,
                    "applied_max_images": 5,
                    "message": None,
                },
                {
                    "name": "portrait_quality",
                    "status": "success",
                    "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                    "output_csv_path": None,
                    "processed_count": 5,
                    "applied_max_images": 5,
                    "message": None,
                },
            ],
        },
    )
    monkeypatch.setattr(
        run_batch,
        "_print_pipeline_summary",
        lambda summary: printed.append(summary),
    )
    monkeypatch.setattr(
        run_batch,
        "_write_pipeline_summary_json",
        lambda summary: Path("/tmp/project/runs/latest/pipeline_summary.json"),
    )

    rc = run_batch.main(
        [
            "--pipeline",
            "nef,portrait_quality",
            "--config",
            "config/config.prod.yaml",
            "--max-images",
            "5",
            "--date",
            "2026-02-17",
        ]
    )

    assert rc == 0
    assert len(printed) == 1
    assert printed[0]["pipeline"] == ["nef", "portrait_quality"]
    assert printed[0]["status"] == "success"


def test_main_processor_mode_still_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []

    class DummyNefProcessor:
        runtime_param_names = ("target_date", "target_dir", "nef_incremental", "nef_max_files", "nef_dry_run")

    monkeypatch.setattr(
        run_batch,
        "resolve_processor",
        lambda spec: DummyNefProcessor,
    )
    monkeypatch.setattr(
        run_batch,
        "run_single_processor",
        lambda processor_spec, ctor_kwargs, exec_kwargs, runtime_overrides: calls.append(
            {
                "processor_spec": processor_spec,
                "ctor_kwargs": dict(ctor_kwargs),
                "exec_kwargs": dict(exec_kwargs),
                "runtime_overrides": dict(runtime_overrides),
            }
        )
        or {
            "name": "nef",
            "status": "success",
            "input_csv_path": None,
            "output_csv_path": None,
            "processed_count": None,
            "applied_max_images": 7,
            "message": None,
        },
    )

    rc = run_batch.main(
        [
            "--processor",
            "nef",
            "--config",
            "config/config.prod.yaml",
            "--max-images",
            "7",
            "--date",
            "2026-02-17",
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["processor_spec"] == "nef"
    assert calls[0]["exec_kwargs"] == {
        "target_date": "2026-02-17",
        "max_images": 7,
    }
    assert calls[0]["runtime_overrides"] == {
        "date": "2026-02-17",
    }
