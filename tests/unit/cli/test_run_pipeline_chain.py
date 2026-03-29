from __future__ import annotations

from typing import Any, Dict, List

from photo_insight.cli import run_batch


def test_run_pipeline_chain_passes_max_images_to_portrait_quality_stage(monkeypatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: Dict[str, Any],
        exec_kwargs: Dict[str, Any],
        runtime_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        calls.append(
            {
                "processor_spec": processor_spec,
                "exec_kwargs": dict(exec_kwargs),
            }
        )

        if processor_spec == "nef":
            return {
                "name": "nef",
                "status": "success",
                "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                "input_csv_path": None,
                "processed_count": 10,
                "applied_max_images": exec_kwargs.get("max_images"),
                "message": None,
                "run_output_dir": "/work/runs/2026-03-15/neffilebatchprocess_xxx",
            }

        return {
            "name": "portrait_quality",
            "status": "success",
            "output_csv_path": None,
            "input_csv_path": exec_kwargs.get("input_csv_path"),
            "processed_count": 10,
            "applied_max_images": exec_kwargs.get("max_images"),
            "message": None,
            "run_output_dir": "/work/runs/2026-03-15/portraitqualitybatchprocessor_xxx",
        }

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    summary = run_batch.run_pipeline_chain(
        stages=["nef", "portrait_quality"],
        ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
        exec_kwargs={"max_images": 10},
        runtime_overrides={"date": "2026-02-17"},
    )

    assert summary["status"] == "success"
    assert len(calls) == 2

    nef_call = calls[0]
    pq_call = calls[1]

    assert nef_call["processor_spec"] == "nef"
    assert nef_call["exec_kwargs"]["max_images"] == 10
    assert nef_call["exec_kwargs"]["target_date"] == "2026-02-17"

    assert pq_call["processor_spec"] == "portrait_quality"
    assert pq_call["exec_kwargs"]["max_images"] == 10
    assert pq_call["exec_kwargs"]["date"] == "2026-02-17"
    assert pq_call["exec_kwargs"]["input_csv_path"].endswith("_raw_exif_data.csv")

    assert summary["stages"][0]["applied_max_images"] == 10
    assert summary["stages"][1]["applied_max_images"] == 10


def test_run_pipeline_chain_stops_when_nef_stage_fails(monkeypatch) -> None:
    calls: List[str] = []

    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: Dict[str, Any],
        exec_kwargs: Dict[str, Any],
        runtime_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        calls.append(processor_spec)

        if processor_spec == "nef":
            return {
                "name": "nef",
                "status": "failed",
                "output_csv_path": None,
                "input_csv_path": None,
                "processed_count": 0,
                "applied_max_images": exec_kwargs.get("max_images"),
                "message": "NEF output CSV path could not be resolved",
                "run_output_dir": "/work/runs/latest",
            }

        return {
            "name": "portrait_quality",
            "status": "success",
            "output_csv_path": None,
            "input_csv_path": exec_kwargs.get("input_csv_path"),
            "processed_count": 0,
            "applied_max_images": exec_kwargs.get("max_images"),
            "message": None,
            "run_output_dir": "/work/runs/latest",
        }

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    summary = run_batch.run_pipeline_chain(
        stages=["nef", "portrait_quality"],
        ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
        exec_kwargs={"max_images": 10},
        runtime_overrides={"date": "2026-02-17"},
    )

    assert summary["status"] == "failed"
    assert calls == ["nef"]
    assert len(summary["stages"]) == 1
    assert summary["stages"][0]["name"] == "nef"
    assert summary["stages"][0]["status"] == "failed"


def test_run_pipeline_chain_includes_duration_sec(monkeypatch) -> None:
    def fake_run_single_processor(
        processor_spec: str,
        ctor_kwargs: Dict[str, Any],
        exec_kwargs: Dict[str, Any],
        runtime_overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        if processor_spec == "nef":
            return {
                "name": "nef",
                "status": "success",
                "output_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
                "input_csv_path": None,
                "processed_count": 10,
                "applied_max_images": exec_kwargs.get("max_images"),
                "message": None,
                "run_output_dir": "/work/runs/latest",
            }

        return {
            "name": "portrait_quality",
            "status": "success",
            "output_csv_path": None,
            "input_csv_path": exec_kwargs.get("input_csv_path"),
            "processed_count": 10,
            "applied_max_images": exec_kwargs.get("max_images"),
            "message": None,
            "run_output_dir": "/work/runs/latest",
        }

    monkeypatch.setattr(run_batch, "run_single_processor", fake_run_single_processor)

    summary = run_batch.run_pipeline_chain(
        stages=["nef", "portrait_quality"],
        ctor_kwargs={"config_path": "config/config.prod.yaml", "max_workers": 2},
        exec_kwargs={"max_images": 10},
        runtime_overrides={"date": "2026-02-17"},
    )

    assert summary["status"] == "success"
    assert "duration_sec" in summary
    assert isinstance(summary["duration_sec"], float)
    assert summary["duration_sec"] >= 0.0
