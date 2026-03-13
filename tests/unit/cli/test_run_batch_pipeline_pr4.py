from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from photo_insight.cli import run_batch


class DummyProcessor:
    def __init__(self, **kwargs: Any) -> None:
        self.ctor_kwargs = kwargs
        self.execute_calls: List[Dict[str, Any]] = []
        self.project_root = Path("/tmp/project")
        self.run_ctx = None

    def execute(self, **kwargs: Any) -> None:
        self.execute_calls.append(dict(kwargs))


class DummyNEFProcessor(DummyProcessor):
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
def injected() -> Dict[str, Any]:
    return {
        "date": "2026-02-17",
    }


def test_build_stage_result_for_nef_returns_pr4_extended_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_csv = "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv"

    monkeypatch.setattr(
        run_batch,
        "_infer_nef_output_csv_path",
        lambda proc, injected: expected_csv,
    )

    proc = DummyNEFProcessor()

    result = run_batch._build_stage_result(
        processor_spec="nef",
        proc=proc,
        injected={"date": "2026-02-17"},
        exec_kwargs={"max_images": 5},
    )

    assert result == {
        "name": "nef",
        "status": "success",
        "input_csv_path": None,
        "output_csv_path": expected_csv,
        "processed_count": 5,
        "applied_max_images": 5,
        "message": None,
    }


def test_build_stage_result_for_portrait_quality_returns_pr4_extended_fields() -> None:
    class DummyPortraitProc:
        processed_count = 5

    result = run_batch._build_stage_result(
        processor_spec="portrait_quality",
        proc=DummyPortraitProc(),
        injected={"date": "2026-02-17"},
        exec_kwargs={
            "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
        },
    )

    assert result == {
        "name": "portrait_quality",
        "status": "success",
        "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
        "output_csv_path": None,
        "processed_count": 5,
        "applied_max_images": None,
        "message": None,
    }


def test_run_pipeline_chain_returns_summary_and_applies_max_images_only_to_first_stage(
    monkeypatch: pytest.MonkeyPatch,
    ctor_kwargs: Dict[str, Any],
    injected: Dict[str, Any],
) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: Dict[str, Any],
        exec_kwargs: Dict[str, Any],
        injected: Dict[str, Any],
    ) -> Dict[str, Any]:
        calls.append(
            {
                "processor_spec": processor_spec,
                "ctor_kwargs": dict(ctor_kwargs),
                "exec_kwargs": dict(exec_kwargs),
                "injected": dict(injected),
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
            }

        raise AssertionError(f"unexpected processor: {processor_spec}")

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    summary = run_batch.run_pipeline_chain(
        stages=["nef", "portrait_quality"],
        ctor_kwargs=ctor_kwargs,
        exec_kwargs={"append_mode": True, "max_images": 5},
        injected=injected,
    )

    assert summary["pipeline"] == ["nef", "portrait_quality"]
    assert summary["status"] == "success"
    assert len(summary["stages"]) == 2

    assert [x["processor_spec"] for x in calls] == ["nef", "portrait_quality"]

    assert calls[0]["exec_kwargs"] == {
        "append_mode": True,
        "max_images": 5,
    }
    assert calls[1]["exec_kwargs"] == {
        "append_mode": True,
        "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
    }

    assert summary["stages"][0]["name"] == "nef"
    assert summary["stages"][0]["applied_max_images"] == 5

    assert summary["stages"][1]["name"] == "portrait_quality"
    assert summary["stages"][1]["input_csv_path"] == "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv"
    assert summary["stages"][1]["applied_max_images"] is None


def test_print_pipeline_summary_outputs_expected_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = {
        "pipeline": ["nef", "portrait_quality"],
        "status": "success",
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
                "applied_max_images": None,
                "message": None,
            },
        ],
    }

    run_batch._print_pipeline_summary(summary)
    captured = capsys.readouterr()

    assert "[pipeline summary]" in captured.out
    assert "pipeline = nef,portrait_quality" in captured.out
    assert "status = success" in captured.out
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
        lambda stages, ctor_kwargs, exec_kwargs, injected: {
            "pipeline": stages,
            "status": "success",
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
                    "applied_max_images": None,
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

    monkeypatch.setattr(
        run_batch,
        "resolve_processor",
        lambda spec: DummyProcessor,
    )
    monkeypatch.setattr(
        run_batch,
        "run_single_processor",
        lambda processor_spec, ctor_kwargs, exec_kwargs, injected: calls.append(
            {
                "processor_spec": processor_spec,
                "ctor_kwargs": dict(ctor_kwargs),
                "exec_kwargs": dict(exec_kwargs),
                "injected": dict(injected),
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
    assert calls[0]["exec_kwargs"]["max_images"] == 7
    assert calls[0]["injected"]["date"] == "2026-02-17"
