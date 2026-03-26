import csv
from unittest.mock import MagicMock, Mock, patch

import pytest

from photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor import (
    PortraitQualityBatchProcessor,
)


@pytest.fixture
def processor(tmp_path):
    # -------------------------
    # Create dummy NEF exif CSV (match production header)
    # -------------------------
    nef_csv_path = tmp_path / "dummy_nef_exif.csv"
    with nef_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["FileName", "Orientation", "BitDepth"])
        w.writerow(["dummy.NEF", "1", "14"])

    with (
        patch("photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor.ImageLoader"),
        patch("photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor.MemoryMonitor"),
        patch(
            "photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor."
            "PortraitQualityBatchProcessor._resolve_nef_input_csv",
            return_value=str(nef_csv_path),
        ),
    ):

        class TestablePortraitQualityBatchProcessor(PortraitQualityBatchProcessor):
            """
            - BaseBatch の契約に寄せるため get_data() は override しない
            - load_data() を実装して Base.get_data() 経由でキャッシュされることを使う
            - setup はテスト用に最小限
            """

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                self.config_manager = MagicMock()
                self.config_manager.get_memory_threshold = MagicMock(return_value=90)

                self.image_data = []
                self.memory_threshold = 90

            def setup(self) -> None:
                self._stop_event.clear()
                self._set_directories_and_files()
                self._load_processed_images()

                self.memory_threshold_exceeded = False
                self.completed_all_batches = False
                self.processed_count_this_run = 0

                self.memory_threshold = self.config_manager.get_memory_threshold(default=90)
                self.logger.info(f"Memory usage threshold set to {self.memory_threshold}% from config.")

            def load_data(self):
                self.image_data = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
                return self.image_data

        proc = TestablePortraitQualityBatchProcessor(
            config_path=None,
            logger=MagicMock(),
            date="2025-01-01",
            batch_size=2,
        )

        picture_root = tmp_path / "picture_root"
        (picture_root / "2025" / "2025-01-01").mkdir(parents=True, exist_ok=True)

        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        proc.config = {
            "batch_size": 2,
            "output_directory": str(out_dir),
            "picture_root": str(picture_root),
        }

        yield proc


def test_setup_loads_data(processor):
    processor.setup()

    data = processor.get_data()
    expected = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    assert data == expected
    assert processor.image_data == expected


def test_setup_sets_memory_threshold(processor):
    processor.config_manager.get_memory_threshold.return_value = 85
    processor.setup()

    assert processor.memory_threshold == 85
    processor.logger.info.assert_any_call("Memory usage threshold set to 85% from config.")


def test_setup_uses_default_memory_threshold_when_not_configured(processor):
    processor.config_manager.get_memory_threshold.return_value = 90
    processor.setup()
    assert processor.memory_threshold == 90


def test_process_batch_skips_all(processor):
    batch = [
        {"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"},
        {"file_name": "img2.jpg", "orientation": "1", "bit_depth": "8"},
    ]
    processor.processed_images = {d["file_name"] for d in batch}

    processor._process_batch_serial = MagicMock()
    processor._process_batch_parallel = MagicMock()
    processor.save_results = MagicMock()
    processor.memory_monitor.get_memory_usage.return_value = 50
    processor.memory_threshold = 90
    processor.max_workers = 1

    result = processor._process_batch(batch)

    processor._process_batch_serial.assert_not_called()
    processor._process_batch_parallel.assert_not_called()
    processor.save_results.assert_not_called()
    assert result == []


def test_process_batch_processes_one(processor, tmp_path, monkeypatch):
    processor.setup()

    processor.result_csv_file = str(tmp_path / "output" / f"evaluation_results_{processor.date}.csv")
    processor.processed_images = set()
    processor.memory_monitor.get_memory_usage.return_value = 50
    processor.memory_threshold = 90
    processor.base_directory = "/tmp/images"
    processor.max_workers = 1
    processor.processed_count_this_run = 0

    mock_result = {"file_name": "img1.jpg", "status": "success", "overall_score": 0.8}
    mock_serial = Mock(return_value=[mock_result])
    mock_mark_many = Mock()
    mock_save = Mock()

    monkeypatch.setattr(processor, "_process_batch_serial", mock_serial)
    monkeypatch.setattr(processor, "_mark_many_as_processed", mock_mark_many)
    monkeypatch.setattr(processor, "save_results", mock_save)
    monkeypatch.setattr(processor.memory_monitor, "log_usage", Mock(return_value=None))

    batch = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    result = processor._process_batch(batch)

    mock_serial.assert_called_once_with(batch)
    mock_save.assert_called_once_with([mock_result], processor.result_csv_file)
    mock_mark_many.assert_called_once_with(["img1.jpg"])

    assert len(result) == 1
    assert result[0]["file_name"] == "img1.jpg"
    assert result[0]["status"] == "success"
    assert result[0]["score"] == 0.8
    assert result[0]["error_type"] is None
    assert processor.processed_count_this_run == 1


def test_execute_full_flow(processor):
    processor.setup = MagicMock()
    processor.process = MagicMock()
    processor.cleanup = MagicMock()
    processor.logger = MagicMock()

    processor.execute()

    processor.setup.assert_called_once()
    processor.process.assert_called_once()
    processor.cleanup.assert_called_once()


def test_process_single_image_marks_and_returns_result(processor, monkeypatch):
    img_info = {
        "file_name": "test1.jpg",
        "orientation": "1",
        "bit_depth": "8",
    }

    processor.base_directory = "/tmp"

    mock_process_image = Mock(
        return_value={
            "file_name": "test1.jpg",
            "status": "success",
            "overall_score": 0.95,
        }
    )

    monkeypatch.setattr(processor, "process_image", mock_process_image)

    result = processor._process_single_image(img_info)

    assert result is not None
    assert result["file_name"] == "test1.jpg"
    assert result["status"] == "success"

    mock_process_image.assert_called_once_with(
        "/tmp/test1.jpg",
        "1",
        "8",
    )


def test_process_batch_sets_memory_threshold_flag(processor):
    processor.result_csv_file = "/tmp/dummy.csv"
    processor.processed_images = set()
    processor.memory_monitor.get_memory_usage.return_value = 95
    processor.logger = MagicMock()
    processor.base_directory = "/tmp/images"
    processor.max_workers = 1

    processor._process_batch_serial = MagicMock()
    processor._process_batch_parallel = MagicMock()
    processor.save_results = MagicMock()

    batch = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    result = processor._process_batch(batch)

    assert processor.memory_threshold_exceeded is True
    processor._process_batch_serial.assert_not_called()
    processor.save_results.assert_not_called()
    assert result == []


def test_process_batch_skips_none_result(processor, monkeypatch):
    processor.result_csv_file = "/tmp/dummy.csv"
    processor.processed_images = set()
    processor.memory_monitor.get_memory_usage.return_value = 50
    processor.base_directory = "/tmp/images"
    processor.max_workers = 1
    processor.processed_count_this_run = 0

    monkeypatch.setattr(processor, "_process_batch_serial", Mock(return_value=[]))
    monkeypatch.setattr(processor, "save_results", Mock())
    monkeypatch.setattr(processor, "_mark_many_as_processed", Mock())
    monkeypatch.setattr(processor.memory_monitor, "log_usage", Mock(return_value=None))

    batch = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    result = processor._process_batch(batch)

    processor.save_results.assert_not_called()
    processor._mark_many_as_processed.assert_not_called()
    assert result == []
    assert processor.processed_count_this_run == 0


def test_process_batch_parallel_invokes_save_results(processor, monkeypatch, tmp_path):
    batch = [
        {"file_name": "a.jpg", "orientation": "1", "bit_depth": "8"},
        {"file_name": "b.jpg", "orientation": "1", "bit_depth": "8"},
    ]

    processor.max_workers = 2
    processor.max_images = None
    processor.result_csv_file = str(tmp_path / "results.csv")
    processor.processed_images_file = str(tmp_path / "processed.txt")
    processor.processed_images = set()
    processor.processed_count_this_run = 0
    processor.memory_threshold_exceeded = False

    monkeypatch.setattr(processor, "_stopping", Mock(return_value=False))
    monkeypatch.setattr(processor, "_check_and_maybe_stop", Mock(return_value=False))
    monkeypatch.setattr(processor, "_should_stop_by_max_images", Mock(return_value=False))
    monkeypatch.setattr(processor.memory_monitor, "log_usage", Mock(return_value=None))

    mock_process_batch_parallel = Mock(
        return_value=[
            {"file_name": "a.jpg", "status": "success", "overall_score": 0.9},
            {"file_name": "b.jpg", "status": "success", "overall_score": 0.8},
        ]
    )
    monkeypatch.setattr(processor, "_process_batch_parallel", mock_process_batch_parallel)

    mock_save = Mock()
    mock_mark_many = Mock()
    monkeypatch.setattr(processor, "save_results", mock_save)
    monkeypatch.setattr(processor, "_mark_many_as_processed", mock_mark_many)

    result = processor._process_batch(batch)

    mock_save.assert_called_once_with(
        [
            {"file_name": "a.jpg", "status": "success", "overall_score": 0.9},
            {"file_name": "b.jpg", "status": "success", "overall_score": 0.8},
        ],
        str(tmp_path / "results.csv"),
    )
    mock_mark_many.assert_called_once_with(["a.jpg", "b.jpg"])

    assert len(result) == 2

    assert result[0]["file_name"] == "a.jpg"
    assert result[0]["status"] == "success"
    assert result[0]["score"] == 0.9
    assert result[0]["error_type"] is None

    assert result[1]["file_name"] == "b.jpg"
    assert result[1]["status"] == "success"
    assert result[1]["score"] == 0.8
    assert result[1]["error_type"] is None

    assert processor.processed_count_this_run == 2


def test_load_data_filters_processed_images(tmp_path):
    class DummyProcessor(PortraitQualityBatchProcessor):
        def load_image_data(self):
            return [
                {"file_name": "a.NEF", "orientation": "Horizontal", "bit_depth": "14"},
                {"file_name": "b.NEF", "orientation": "Vertical", "bit_depth": "14"},
                {"file_name": "c.NEF", "orientation": "Horizontal", "bit_depth": "12"},
            ]

    p = DummyProcessor(
        config_path=None,
        logger=MagicMock(),
        date="2025-01-01",
        batch_size=2,
    )
    p.config = {
        "batch_size": 2,
        "output_directory": str(tmp_path / "output"),
        "picture_root": str(tmp_path / "picture_root"),
    }

    p.processed_images = {"b.NEF"}

    result = p.load_data()
    file_names = [d["file_name"] for d in result]

    assert set(file_names) == {"a.NEF", "c.NEF"}


def test_process_batch_serial_stops_before_submitting_over_max_images(processor, monkeypatch):
    batch = [
        {"file_name": "a.jpg", "orientation": "1", "bit_depth": "8"},
        {"file_name": "b.jpg", "orientation": "1", "bit_depth": "8"},
        {"file_name": "c.jpg", "orientation": "1", "bit_depth": "8"},
    ]

    processor.max_images = 2
    processor.processed_count_this_run = 0
    processor.memory_monitor.get_memory_usage.return_value = 50

    mock_single = Mock(
        side_effect=[
            {"file_name": "a.jpg", "status": "success"},
            {"file_name": "b.jpg", "status": "success"},
        ]
    )
    monkeypatch.setattr(processor, "_process_single_image", mock_single)
    monkeypatch.setattr(processor, "_check_and_maybe_stop", Mock(return_value=False))
    monkeypatch.setattr(processor, "request_stop", Mock())

    results = processor._process_batch_serial(batch)

    assert len(results) == 2
    assert [r["file_name"] for r in results] == ["a.jpg", "b.jpg"]
    assert mock_single.call_count == 2
    processor.request_stop.assert_called_once_with("max_images_limit")


def test_process_batch_serial_respects_already_processed_count(processor, monkeypatch):
    batch = [
        {"file_name": "a.jpg", "orientation": "1", "bit_depth": "8"},
        {"file_name": "b.jpg", "orientation": "1", "bit_depth": "8"},
    ]

    processor.max_images = 2
    processor.processed_count_this_run = 1
    processor.memory_monitor.get_memory_usage.return_value = 50

    mock_single = Mock(return_value={"file_name": "a.jpg", "status": "success"})
    monkeypatch.setattr(processor, "_process_single_image", mock_single)
    monkeypatch.setattr(processor, "_check_and_maybe_stop", Mock(return_value=False))
    monkeypatch.setattr(processor, "request_stop", Mock())

    results = processor._process_batch_serial(batch)

    assert len(results) == 1
    assert results[0]["file_name"] == "a.jpg"
    assert mock_single.call_count == 1
    processor.request_stop.assert_called_once_with("max_images_limit")


def test_process_batch_parallel_stops_before_submit_when_max_images_reached(processor, monkeypatch):
    batch = [
        {"file_name": "a.jpg", "orientation": "1", "bit_depth": "8"},
        {"file_name": "b.jpg", "orientation": "1", "bit_depth": "8"},
        {"file_name": "c.jpg", "orientation": "1", "bit_depth": "8"},
    ]

    processor.max_workers = 3
    processor.max_images = 2
    processor.processed_count_this_run = 0
    processor.memory_monitor.get_memory_usage.return_value = 50

    submitted = []

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def cancel(self):
            return False

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, img_info):
            submitted.append(img_info["file_name"])
            return DummyFuture({"file_name": img_info["file_name"], "status": "success"})

    monkeypatch.setattr(
        "photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor.ThreadPoolExecutor",
        DummyExecutor,
    )
    monkeypatch.setattr(
        "photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor.as_completed",
        lambda futures: futures,
    )
    monkeypatch.setattr(processor, "_check_and_maybe_stop", Mock(return_value=False))
    monkeypatch.setattr(processor, "request_stop", Mock())

    results = processor._process_batch_parallel(batch)

    assert submitted == ["a.jpg", "b.jpg"]
    assert [r["file_name"] for r in results] == ["a.jpg", "b.jpg"]
    processor.request_stop.assert_called_once_with("max_images_limit")


def test_process_batch_parallel_respects_already_processed_count(processor, monkeypatch):
    batch = [
        {"file_name": "a.jpg", "orientation": "1", "bit_depth": "8"},
        {"file_name": "b.jpg", "orientation": "1", "bit_depth": "8"},
    ]

    processor.max_workers = 2
    processor.max_images = 2
    processor.processed_count_this_run = 1
    processor.memory_monitor.get_memory_usage.return_value = 50

    submitted = []

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def cancel(self):
            return False

    class DummyExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, img_info):
            submitted.append(img_info["file_name"])
            return DummyFuture({"file_name": img_info["file_name"], "status": "success"})

    monkeypatch.setattr(
        "photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor.ThreadPoolExecutor",
        DummyExecutor,
    )
    monkeypatch.setattr(
        "photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor.as_completed",
        lambda futures: futures,
    )
    monkeypatch.setattr(processor, "_check_and_maybe_stop", Mock(return_value=False))
    monkeypatch.setattr(processor, "request_stop", Mock())

    results = processor._process_batch_parallel(batch)

    assert submitted == ["a.jpg"]
    assert [r["file_name"] for r in results] == ["a.jpg"]
    processor.request_stop.assert_called_once_with("max_images_limit")


def test_process_batch_returns_empty_when_max_images_zero(processor, monkeypatch):
    processor.max_images = 0
    processor.processed_count_this_run = 0
    processor.max_workers = 1
    processor.result_csv_file = "/tmp/dummy.csv"
    processor.processed_images = set()

    monkeypatch.setattr(processor, "_stopping", Mock(return_value=False))
    monkeypatch.setattr(processor, "_check_and_maybe_stop", Mock(return_value=False))
    monkeypatch.setattr(processor.memory_monitor, "log_usage", Mock(return_value=None))

    processor.save_results = Mock()
    processor._mark_many_as_processed = Mock()

    batch = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    result = processor._process_batch(batch)

    assert result == []
    processor.save_results.assert_not_called()
    processor._mark_many_as_processed.assert_not_called()
    assert processor.get_stop_reason() == "max_images_limit"
