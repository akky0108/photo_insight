from __future__ import annotations

from photo_insight.cli.run_batch import _build_stage_exec_kwargs, resolve_processor


def test_build_stage_exec_kwargs_for_nef_maps_date_to_target_date() -> None:
    processor_cls = resolve_processor("nef")

    common_exec_kwargs = {"max_images": 10}
    runtime_overrides = {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026/2026-02-17",
        "nef_incremental": True,
        "nef_max_files": 25,
    }

    got = _build_stage_exec_kwargs(
        processor_spec="nef",
        processor_cls=processor_cls,
        common_exec_kwargs=common_exec_kwargs,
        runtime_overrides=runtime_overrides,
    )

    assert got["target_date"] == "2026-02-17"
    assert got["target_dir"] == "/work/input/2026/2026-02-17"
    assert got["nef_incremental"] is True
    assert got["nef_max_files"] == 25
    assert "date" not in got


def test_build_stage_exec_kwargs_for_nef_filters_unknown_runtime_keys() -> None:
    processor_cls = resolve_processor("nef")

    common_exec_kwargs = {
        "max_images": 10,
        "input_csv_path": "/tmp/upstream.csv",
    }
    runtime_overrides = {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026/2026-02-17",
        "nef_incremental": True,
    }

    got = _build_stage_exec_kwargs(
        processor_spec="nef",
        processor_cls=processor_cls,
        common_exec_kwargs=common_exec_kwargs,
        runtime_overrides=runtime_overrides,
    )

    # passthrough key は残る
    assert got["max_images"] == 10
    assert got["input_csv_path"] == "/tmp/upstream.csv"

    # NEF 向け runtime は残る
    assert got["target_date"] == "2026-02-17"
    assert got["target_dir"] == "/work/input/2026/2026-02-17"
    assert got["nef_incremental"] is True


def test_build_stage_exec_kwargs_for_portrait_quality_uses_date() -> None:
    processor_cls = resolve_processor("portrait_quality")

    common_exec_kwargs = {"max_images": 10}
    runtime_overrides = {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026/2026-02-17",
        "nef_incremental": True,
        "nef_max_files": 25,
    }

    got = _build_stage_exec_kwargs(
        processor_spec="portrait_quality",
        processor_cls=processor_cls,
        common_exec_kwargs=common_exec_kwargs,
        runtime_overrides=runtime_overrides,
    )

    assert got["date"] == "2026-02-17"
    assert got["target_dir"] == "/work/input/2026/2026-02-17"
    assert got["max_images"] == 10

    assert "target_date" not in got
    assert "nef_incremental" not in got
    assert "nef_max_files" not in got


def test_build_stage_exec_kwargs_for_portrait_quality_keeps_input_csv_path() -> None:
    processor_cls = resolve_processor("portrait_quality")

    common_exec_kwargs = {
        "max_images": 10,
        "input_csv_path": "/work/runs/latest/nef/2026-02-17/2026-02-17_raw_exif_data.csv",
    }
    runtime_overrides = {
        "date": "2026-02-17",
    }

    got = _build_stage_exec_kwargs(
        processor_spec="portrait_quality",
        processor_cls=processor_cls,
        common_exec_kwargs=common_exec_kwargs,
        runtime_overrides=runtime_overrides,
    )

    assert got["date"] == "2026-02-17"
    assert got["max_images"] == 10
    assert got["input_csv_path"].endswith("_raw_exif_data.csv")


def test_build_stage_exec_kwargs_for_dotted_processor_falls_back_to_runtime_param_names() -> None:
    processor_cls = resolve_processor("photo_insight.pipelines.portrait_quality:PortraitQualityBatchProcessor")

    common_exec_kwargs = {"max_images": 10}
    runtime_overrides = {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026/2026-02-17",
    }

    got = _build_stage_exec_kwargs(
        processor_spec="photo_insight.pipelines.portrait_quality:PortraitQualityBatchProcessor",
        processor_cls=processor_cls,
        common_exec_kwargs=common_exec_kwargs,
        runtime_overrides=runtime_overrides,
    )

    assert got["date"] == "2026-02-17"
    assert got["target_dir"] == "/work/input/2026/2026-02-17"
    assert got["max_images"] == 10
