import pytest
from unittest.mock import MagicMock, patch
from portrait_quality_batch_processor import PortraitQualityBatchProcessor


@pytest.fixture
def processor(tmp_path):
    with patch("portrait_quality_batch_processor.ImageLoader"):
        with patch("portrait_quality_batch_processor.MemoryMonitor"):

            class TestablePortraitQualityBatchProcessor(PortraitQualityBatchProcessor):
                def get_data(self):
                    return []

            proc = TestablePortraitQualityBatchProcessor(
                config_path=None, logger=MagicMock(), date="2025-01-01", batch_size=2
            )
            proc.config = {
                "batch_size": 2,
                "output_directory": str(tmp_path / "output"),
                "base_directory_root": str(tmp_path / "base"),
            }
            return proc


def test_setup_loads_data(processor):
    with patch(
        "portrait_quality_batch_processor.os.path.exists", return_value=False
    ), patch("builtins.open", create=True), patch.object(
        processor,
        "load_image_data",
        return_value=[{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}],
    ):

        processor.setup()
        assert processor.data == [
            {"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}
        ]
        assert processor.image_data == processor.data


def test_process_batch_skips_all(processor):
    processor.processed_images = {"img1.jpg", "img2.jpg"}
    processor.logger = MagicMock()
    processor.memory_monitor.get_memory_usage.return_value = 50

    processor._process_batch(
        [
            {"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"},
            {"file_name": "img2.jpg", "orientation": "1", "bit_depth": "8"},
        ]
    )

    processor.logger.info.assert_any_call(
        "All images in this batch are already processed. Skipping."
    )


def test_process_batch_processes_one(processor, tmp_path):
    processor.setup()
    processor.result_csv_file = (
        tmp_path / "output" / f"evaluation_results_{processor.date}.csv"
    )
    processor.processed_images = set()
    processor.memory_monitor.get_memory_usage.return_value = 50

    mock_result = {"file_name": "img1.jpg", "sharpness_score": 0.8}
    processor.process_image = MagicMock(return_value=mock_result)
    processor._mark_as_processed = MagicMock()
    processor.save_results = MagicMock()

    batch = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    processor.base_directory = "/tmp/images"
    processor.max_workers = 1

    processor._process_batch(batch)

    processor.process_image.assert_called_once()
    processor.save_results.assert_called_once_with(
        [mock_result], processor.result_csv_file
    )


def test_execute_full_flow(processor):
    processor.setup = MagicMock()
    processor.cleanup = MagicMock()
    processor.logger = MagicMock()
    processor.data = [
        {"file_name": f"img{i}.jpg", "orientation": "1", "bit_depth": "8"}
        for i in range(4)
    ]
    processor.processed_images = set()
    processor.memory_threshold_exceeded = False
    processor._process_batch = MagicMock()

    processor.execute()

    assert processor._process_batch.call_count == 2
    processor.cleanup.assert_called_once()


# --- 以下追加分 ---


def test_process_single_image_marks_and_returns_result(processor):
    processor.process_image = MagicMock(
        side_effect=lambda path, orientation, bit_depth: {
            "file_name": "img1.jpg",
            "sharpness_score": 0.9,
        }
    )
    processor._mark_as_processed = MagicMock()
    processor.base_directory = "/dummy/path"

    img_info = {"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}
    result = processor._process_single_image(img_info)

    assert result == {"file_name": "img1.jpg", "sharpness_score": 0.9}
    processor._mark_as_processed.assert_called_once_with("img1.jpg")


def test_process_batch_sets_memory_threshold_flag(processor):
    processor.result_csv_file = "/tmp/dummy.csv"
    processor.processed_images = set()
    processor.memory_monitor.get_memory_usage.return_value = 95
    processor.logger = MagicMock()
    processor.base_directory = "/tmp/images"
    processor.max_workers = 1

    processor.process_image = MagicMock(return_value={"file_name": "img1.jpg"})
    processor._mark_as_processed = MagicMock()
    processor.save_results = MagicMock()

    batch = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    processor._process_batch(batch)

    assert processor.memory_threshold_exceeded is True


def test_process_batch_skips_none_result(processor):
    processor.result_csv_file = "/tmp/dummy.csv"
    processor.processed_images = set()
    processor.memory_monitor.get_memory_usage.return_value = 50
    processor.base_directory = "/tmp/images"
    processor.max_workers = 1

    processor.process_image = MagicMock(return_value=None)
    processor._mark_as_processed = MagicMock()
    processor.save_results = MagicMock()

    batch = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    processor._process_batch(batch)

    processor.save_results.assert_not_called()


def test_process_batch_parallel_invokes_save_results(processor, tmp_path):
    processor.result_csv_file = tmp_path / "output.csv"
    processor.processed_images = set()
    processor.memory_monitor.get_memory_usage.return_value = 50
    processor.logger = MagicMock()

    processor.max_workers = 2
    processor.base_directory = "/tmp/images"
    processor.process_image = MagicMock(
        return_value={"file_name": "img1.jpg", "sharpness_score": 0.9}
    )
    processor._mark_as_processed = MagicMock()
    processor.save_results = MagicMock()

    batch = [{"file_name": "img1.jpg", "orientation": "1", "bit_depth": "8"}]
    processor._process_batch(batch)

    processor.save_results.assert_called_once_with(
        [{"file_name": "img1.jpg", "sharpness_score": 0.9}], processor.result_csv_file
    )
