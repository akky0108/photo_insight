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


def test_run_single_processor_resolves_and_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: Dict[str, Any] = {}

    class LocalProcessor(DummyProcessor):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            created["instance"] = self

    def fake_resolve_processor(spec: str) -> type[LocalProcessor]:
        assert spec == "nef"
        return LocalProcessor

    def fake_build_stage_result(
        processor_spec: str,
        proc: Any,
        injected: dict[str, Any],
        exec_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert processor_spec == "nef"
        assert proc is created["instance"]
        assert injected == {"date": "2026-02-17"}
        assert exec_kwargs == {"append_mode": True}
        return {
            "name": "nef",
            "status": "success",
            "input_csv_path": None,
            "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
            "processed_count": None,
            "applied_max_images": None,
            "message": None,
        }

    monkeypatch.setattr(run_batch, "resolve_processor", fake_resolve_processor)
    monkeypatch.setattr(run_batch, "_build_stage_result", fake_build_stage_result)

    result = run_batch.run_single_processor(
        processor_spec="nef",
        ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
        exec_kwargs={"append_mode": True},
        injected={"date": "2026-02-17"},
    )

    proc = created["instance"]
    assert proc.ctor_kwargs == {
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
    }
    assert proc.date == "2026-02-17"
    assert proc.target_date == "2026-02-17"
    assert proc.execute_calls == [{"append_mode": True}]

    assert result == {
        "name": "nef",
        "status": "success",
        "input_csv_path": None,
        "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
        "processed_count": None,
        "applied_max_images": None,
        "message": None,
    }


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
                "processed_count": None,
                "applied_max_images": None,
                "message": None,
                "run_output_dir": "/tmp/project/runs/latest",
            }

        return {
            "name": "portrait_quality",
            "status": "success",
            "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
            "output_csv_path": None,
            "processed_count": None,
            "applied_max_images": None,
            "message": None,
            "run_output_dir": "/tmp/project/runs/latest",
        }

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    summary = run_batch.run_pipeline_chain(
        stages=["nef", "portrait_quality"],
        ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
        exec_kwargs={"append_mode": True},
        injected={"date": "2026-02-17"},
    )

    assert [x["processor_spec"] for x in calls] == ["nef", "portrait_quality"]

    assert calls[0]["exec_kwargs"] == {
        "append_mode": True,
    }
    assert calls[1]["exec_kwargs"] == {
        "append_mode": True,
        "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
    }

    assert summary["summary_version"] == 1
    assert summary["pipeline"] == ["nef", "portrait_quality"]
    assert summary["status"] == "success"
    assert summary["run_context"] == {
        "date": "2026-02-17",
        "target_dir": None,
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
        "max_images": None,
        "config_env": None,
        "config_paths": None,
    }
    assert summary["stages"] == [
        {
            "name": "nef",
            "status": "success",
            "input_csv_path": None,
            "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
            "processed_count": None,
            "applied_max_images": None,
            "message": None,
            "run_output_dir": "/tmp/project/runs/latest",
        },
        {
            "name": "portrait_quality",
            "status": "success",
            "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
            "output_csv_path": None,
            "processed_count": None,
            "applied_max_images": None,
            "message": None,
            "run_output_dir": "/tmp/project/runs/latest",
        },
    ]


def test_main_pipeline_execution_calls_run_pipeline_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}
    printed: List[Dict[str, Any]] = []

    def fake_run_pipeline_chain(
        stages: list[str],
        ctor_kwargs: dict[str, Any],
        exec_kwargs: dict[str, Any],
        injected: dict[str, Any],
    ) -> dict[str, Any]:
        captured["stages"] = stages
        captured["ctor_kwargs"] = dict(ctor_kwargs)
        captured["exec_kwargs"] = dict(exec_kwargs)
        captured["injected"] = dict(injected)
        return {
            "pipeline": stages,
            "status": "success",
            "stages": [
                {
                    "name": "nef",
                    "status": "success",
                    "input_csv_path": None,
                    "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                    "processed_count": None,
                    "applied_max_images": None,
                    "message": None,
                },
                {
                    "name": "portrait_quality",
                    "status": "success",
                    "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                    "output_csv_path": None,
                    "processed_count": None,
                    "applied_max_images": None,
                    "message": None,
                },
            ],
        }

    monkeypatch.setattr(run_batch, "run_pipeline_chain", fake_run_pipeline_chain)
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
            "--max-workers",
            "2",
            "--date",
            "2026-02-17",
            "--append-mode",
            "true",
        ]
    )

    assert rc == 0
    assert captured["stages"] == ["nef", "portrait_quality"]
    assert captured["ctor_kwargs"] == {
        "config_path": "config/config.prod.yaml",
        "max_workers": 2,
    }
    assert captured["exec_kwargs"] == {
        "append_mode": True,
    }
    assert captured["injected"] == {
        "date": "2026-02-17",
    }

    assert len(printed) == 1
    assert printed[0]["pipeline"] == ["nef", "portrait_quality"]
    assert printed[0]["status"] == "success"
