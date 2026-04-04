from pathlib import Path

from photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor import (
    PortraitQualityBatchProcessor,
)


def test_process_single_image_returns_result(monkeypatch, tmp_path):
    p = PortraitQualityBatchProcessor(
        config_path="config/config.test.yaml",
        max_workers=1,
        date="2026-02-17",
        max_images=1,
    )

    p.base_directory = str(tmp_path)
    (tmp_path / "DSC_7477.NEF").write_bytes(b"dummy")

    monkeypatch.setattr(
        p,
        "process_image",
        lambda file_name, orientation, bit_depth: {
            "file_name": Path(file_name).name,
            "status": "success",
            "overall_score": 0.8,
        },
    )

    result = p._process_single_image(
        {
            "file_name": "DSC_7477.NEF",
            "orientation": "1",
            "bit_depth": "14",
        }
    )

    assert result is not None
    assert result["status"] == "success"


def test_process_batch_serial_collects_result(monkeypatch, tmp_path):
    p = PortraitQualityBatchProcessor(
        config_path="config/config.test.yaml",
        max_workers=1,
        date="2026-02-17",
        max_images=1,
    )

    p.base_directory = str(tmp_path)
    p.processed_count_this_run = 0
    p._stop_event.clear()
    (tmp_path / "DSC_7477.NEF").write_bytes(b"dummy")

    monkeypatch.setattr(
        p,
        "_process_single_image",
        lambda img_info: {
            "file_name": img_info["file_name"],
            "status": "success",
            "overall_score": 0.8,
        },
    )

    results = p._process_batch_serial(
        [
            {
                "file_name": "DSC_7477.NEF",
                "orientation": "1",
                "bit_depth": "14",
            }
        ]
    )

    assert len(results) == 1
    assert results[0]["status"] == "success"


def test_process_batch_returns_mini_result_and_updates_count(monkeypatch, tmp_path):
    p = PortraitQualityBatchProcessor(
        config_path="config/config.test.yaml",
        max_workers=1,
        date="2026-02-17",
        max_images=1,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    p.base_directory = str(tmp_path)
    p.output_directory = str(out_dir)
    p.result_csv_file = str(out_dir / "evaluation_results_2026-02-17.csv")
    p.processed_images_file = str(out_dir / "processed_images_2026-02-17.txt")
    p.processed_count_this_run = 0
    p.processed_images = set()
    p._stop_event.clear()

    monkeypatch.setattr(
        p,
        "_process_batch_serial",
        lambda batch: [
            {
                "file_name": "DSC_7477.NEF",
                "status": "success",
                "overall_score": 0.8,
            }
        ],
    )

    results = p._process_batch(
        [
            {
                "file_name": "DSC_7477.NEF",
                "orientation": "1",
                "bit_depth": "14",
            }
        ]
    )

    assert len(results) == 1
    assert p.processed_count_this_run == 1
    assert Path(p.result_csv_file).exists()
    assert Path(p.processed_images_file).exists()


def test_process_batch_no_result_does_not_mark_processed(monkeypatch, tmp_path):
    p = PortraitQualityBatchProcessor(
        config_path="config/config.test.yaml",
        max_workers=1,
        date="2026-02-17",
        max_images=1,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    p.base_directory = str(tmp_path)
    p.output_directory = str(out_dir)
    p.result_csv_file = str(out_dir / "evaluation_results_2026-02-17.csv")
    p.processed_images_file = str(out_dir / "processed_images_2026-02-17.txt")
    p.processed_count_this_run = 0
    p.processed_images = set()
    p._stop_event.clear()

    monkeypatch.setattr(
        p,
        "_process_batch_serial",
        lambda batch: [
            {
                "file_name": "DSC_7477.NEF",
                "status": "no_result",
                "overall_score": None,
                "error_type": "empty_evaluation",
            }
        ],
    )

    results = p._process_batch(
        [
            {
                "file_name": "DSC_7477.NEF",
                "orientation": "1",
                "bit_depth": "14",
            }
        ]
    )

    assert len(results) == 1
    assert results[0]["status"] == "no_result"
    assert p.processed_count_this_run == 1


def test_process_batch_evaluate_error_does_not_mark_processed(monkeypatch, tmp_path):
    p = PortraitQualityBatchProcessor(
        config_path="config/config.test.yaml",
        max_workers=1,
        date="2026-02-17",
        max_images=1,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    p.base_directory = str(tmp_path)
    p.output_directory = str(out_dir)
    p.result_csv_file = str(out_dir / "evaluation_results_2026-02-17.csv")
    p.processed_images_file = str(out_dir / "processed_images_2026-02-17.txt")
    p.processed_count_this_run = 0
    p.processed_images = set()
    p._stop_event.clear()

    monkeypatch.setattr(
        p,
        "_process_batch_serial",
        lambda batch: [
            {
                "file_name": "DSC_7477.NEF",
                "status": "evaluate_error",
                "overall_score": None,
                "error_type": "RuntimeError",
            }
        ],
    )

    results = p._process_batch(
        [
            {
                "file_name": "DSC_7477.NEF",
                "orientation": "1",
                "bit_depth": "14",
            }
        ]
    )

    assert len(results) == 1
    assert results[0]["status"] == "evaluate_error"
    assert p.processed_count_this_run == 1
