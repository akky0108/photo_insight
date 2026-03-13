from __future__ import annotations

import json
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
        self.run_ctx = type("RunCtx", (), {"out_dir": "/tmp/project/runs/2026-03-13/run_001"})()


class DummyPortraitProcessor(DummyProcessor):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.processed_count = 5
        self.run_ctx = type("RunCtx", (), {"out_dir": "/tmp/project/runs/2026-03-13/run_001"})()


@pytest.fixture
def ctor_kwargs() -> Dict[str, Any]:
    return {
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
        "config_env": None,
    }


@pytest.fixture
def injected() -> Dict[str, Any]:
    return {
        "date": "2026-02-17",
    }


def test_infer_run_output_dir_prefers_run_ctx_out_dir() -> None:
    proc = DummyNEFProcessor()
    result = run_batch._infer_run_output_dir(proc)
    assert result == "/tmp/project/runs/2026-03-13/run_001"


def test_infer_run_output_dir_falls_back_to_project_root_runs_latest() -> None:
    proc = DummyProcessor()
    result = run_batch._infer_run_output_dir(proc)
    assert result == "/tmp/project/runs/latest"


def test_build_stage_result_includes_run_output_dir_for_nef(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_csv = "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv"
    proc = DummyNEFProcessor()

    monkeypatch.setattr(
        run_batch,
        "_infer_nef_output_csv_path",
        lambda proc, injected: expected_csv,
    )

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
        "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
    }


def test_build_pipeline_run_context_returns_expected_fields(
    ctor_kwargs: Dict[str, Any],
    injected: Dict[str, Any],
) -> None:
    result = run_batch._build_pipeline_run_context(
        ctor_kwargs=ctor_kwargs,
        exec_kwargs={"max_images": 10, "append_mode": True},
        injected=injected,
    )

    assert result == {
        "date": "2026-02-17",
        "target_dir": None,
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
        "max_images": 10,
        "config_env": None,
        "config_paths": None,
    }


def test_build_pipeline_summary_includes_summary_version_and_run_context(
    ctor_kwargs: Dict[str, Any],
    injected: Dict[str, Any],
) -> None:
    stage_results = [
        {
            "name": "nef",
            "status": "success",
            "input_csv_path": None,
            "output_csv_path": "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
            "processed_count": 5,
            "applied_max_images": 5,
            "message": None,
            "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
        },
        {
            "name": "portrait_quality",
            "status": "success",
            "input_csv_path": "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
            "output_csv_path": None,
            "processed_count": 5,
            "applied_max_images": None,
            "message": None,
            "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
        },
    ]

    summary = run_batch._build_pipeline_summary(
        stages=["nef", "portrait_quality"],
        stage_results=stage_results,
        ctor_kwargs=ctor_kwargs,
        exec_kwargs={"max_images": 5},
        injected=injected,
    )

    assert summary == {
        "summary_version": 1,
        "pipeline": ["nef", "portrait_quality"],
        "status": "success",
        "run_context": {
            "date": "2026-02-17",
            "target_dir": None,
            "config_path": "config/config.prod.yaml",
            "max_workers": 2,
            "max_images": 5,
            "config_env": None,
            "config_paths": None,
        },
        "stages": stage_results,
    }


def test_resolve_pipeline_summary_output_path_uses_first_stage_run_output_dir() -> None:
    summary = {
        "summary_version": 1,
        "pipeline": ["nef", "portrait_quality"],
        "status": "success",
        "run_context": {},
        "stages": [
            {
                "name": "nef",
                "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
            },
            {
                "name": "portrait_quality",
                "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
            },
        ],
    }

    path = run_batch._resolve_pipeline_summary_output_path(summary)
    assert path == Path("/tmp/project/runs/2026-03-13/run_001/pipeline_summary.json")


def test_write_pipeline_summary_json_writes_expected_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "runs" / "2026-03-13" / "run_001" / "pipeline_summary.json"
    summary = {
        "summary_version": 1,
        "pipeline": ["nef", "portrait_quality"],
        "status": "success",
        "run_context": {
            "date": "2026-02-17",
            "target_dir": None,
            "config_path": "config/config.prod.yaml",
            "max_workers": 2,
            "max_images": 5,
            "config_env": None,
            "config_paths": None,
        },
        "stages": [
            {
                "name": "nef",
                "status": "success",
                "input_csv_path": None,
                "output_csv_path": "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                "processed_count": 5,
                "applied_max_images": 5,
                "message": None,
                "run_output_dir": str(tmp_path / "runs" / "2026-03-13" / "run_001"),
            }
        ],
    }

    monkeypatch.setattr(
        run_batch,
        "_resolve_pipeline_summary_output_path",
        lambda summary: output_path,
    )

    saved_path = run_batch._write_pipeline_summary_json(summary)

    assert saved_path == output_path
    assert output_path.exists()

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == summary


def test_run_pipeline_chain_returns_summary_with_run_context_and_run_output_dir(
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
                "output_csv_path": "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                "processed_count": 5,
                "applied_max_images": exec_kwargs.get("max_images"),
                "message": None,
                "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
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
                "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
            }

        raise AssertionError(f"unexpected processor: {processor_spec}")

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    summary = run_batch.run_pipeline_chain(
        stages=["nef", "portrait_quality"],
        ctor_kwargs=ctor_kwargs,
        exec_kwargs={"append_mode": True, "max_images": 5},
        injected=injected,
    )

    assert summary["summary_version"] == 1
    assert summary["pipeline"] == ["nef", "portrait_quality"]
    assert summary["status"] == "success"
    assert summary["run_context"] == {
        "date": "2026-02-17",
        "target_dir": None,
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
        "max_images": 5,
        "config_env": None,
        "config_paths": None,
    }

    assert [x["processor_spec"] for x in calls] == ["nef", "portrait_quality"]
    assert calls[0]["exec_kwargs"] == {
        "append_mode": True,
        "max_images": 5,
    }
    assert calls[1]["exec_kwargs"] == {
        "append_mode": True,
        "input_csv_path": "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
    }

    assert summary["stages"][0]["run_output_dir"] == "/tmp/project/runs/2026-03-13/run_001"
    assert summary["stages"][1]["run_output_dir"] == "/tmp/project/runs/2026-03-13/run_001"


def test_print_pipeline_summary_includes_run_context_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = {
        "summary_version": 1,
        "pipeline": ["nef", "portrait_quality"],
        "status": "success",
        "run_context": {
            "date": "2026-02-17",
            "target_dir": "/work/input/2026-02-17",
            "config_path": "config/config.prod.yaml",
            "max_workers": 2,
            "max_images": 5,
            "config_env": None,
            "config_paths": None,
        },
        "stages": [
            {
                "name": "nef",
                "status": "success",
                "input_csv_path": None,
                "output_csv_path": "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                "processed_count": 5,
                "applied_max_images": 5,
                "message": None,
                "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
            }
        ],
    }

    run_batch._print_pipeline_summary(summary)
    captured = capsys.readouterr()

    assert "[pipeline summary]" in captured.out
    assert "pipeline = nef,portrait_quality" in captured.out
    assert "status = success" in captured.out
    assert "date = 2026-02-17" in captured.out
    assert "target_dir = /work/input/2026-02-17" in captured.out
    assert "max_images = 5" in captured.out
    assert "[stage] nef" in captured.out


def test_main_pipeline_writes_summary_json_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: List[Dict[str, Any]] = []
    saved: List[Dict[str, Any]] = []

    monkeypatch.setattr(
        run_batch,
        "run_pipeline_chain",
        lambda stages, ctor_kwargs, exec_kwargs, injected: {
            "summary_version": 1,
            "pipeline": stages,
            "status": "success",
            "run_context": {
                "date": "2026-02-17",
                "target_dir": None,
                "config_path": "config/config.prod.yaml",
                "max_workers": 2,
                "max_images": 5,
                "config_env": None,
                "config_paths": None,
            },
            "stages": [
                {
                    "name": "nef",
                    "status": "success",
                    "input_csv_path": None,
                    "output_csv_path": "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                    "processed_count": 5,
                    "applied_max_images": 5,
                    "message": None,
                    "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
                },
                {
                    "name": "portrait_quality",
                    "status": "success",
                    "input_csv_path": "/tmp/project/runs/2026-03-13/run_001/artifacts/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                    "output_csv_path": None,
                    "processed_count": 5,
                    "applied_max_images": None,
                    "message": None,
                    "run_output_dir": "/tmp/project/runs/2026-03-13/run_001",
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
        lambda summary: saved.append(summary) or Path("/tmp/project/runs/2026-03-13/run_001/pipeline_summary.json"),
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
    assert len(saved) == 1
    assert printed[0]["summary_version"] == 1
    assert saved[0]["pipeline"] == ["nef", "portrait_quality"]


def test_main_pipeline_dry_run_does_not_write_summary_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"run_pipeline_chain": 0, "print_summary": 0, "write_json": 0}

    monkeypatch.setattr(
        run_batch,
        "run_pipeline_chain",
        lambda stages, ctor_kwargs, exec_kwargs, injected: called.__setitem__(
            "run_pipeline_chain", called["run_pipeline_chain"] + 1
        ),
    )
    monkeypatch.setattr(
        run_batch,
        "_print_pipeline_summary",
        lambda summary: called.__setitem__("print_summary", called["print_summary"] + 1),
    )
    monkeypatch.setattr(
        run_batch,
        "_write_pipeline_summary_json",
        lambda summary: called.__setitem__("write_json", called["write_json"] + 1),
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
            "--dry-run",
        ]
    )

    assert rc == 0
    assert called["run_pipeline_chain"] == 0
    assert called["print_summary"] == 0
    assert called["write_json"] == 0
