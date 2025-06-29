import csv
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor

from batch_processor.nef_batch_process import NEFFileBatchProcess


# --- テスト専用のダミークラス定義（抽象メソッドを埋める） ---
# class DummyNEFFileBatchProcess(NEFFileBatchProcess):
#     def _process_batch(self, batch: list[dict]) -> None:
#         # 親クラスの処理を呼び出す
#         super()._process_batch(batch)

# --- 共通フィクスチャ ---
@pytest.fixture
def dummy_processor(tmp_path):
    processor = NEFFileBatchProcess(config_path=None)
    processor.project_root = tmp_path
    processor.temp_dir = tmp_path / "temp"
    processor.temp_dir.mkdir()
    processor.logger = MagicMock()
    return processor


# --- テスト本体 ---
def test_get_target_subdirectories(tmp_path, dummy_processor):
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_file = old_dir / "dummy.txt"
    old_file.write_text("dummy")

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_file = new_dir / "dummy.txt"
    new_file.write_text("dummy")

    import os, time
    old_timestamp = time.time() - (60 * 60 * 24 * 365)
    os.utime(old_dir, (old_timestamp, old_timestamp))
    os.utime(old_file, (old_timestamp, old_timestamp))

    dummy_processor.target_date = datetime.now() - timedelta(minutes=1)

    subdirs = dummy_processor.get_target_subdirectories(tmp_path)

    assert new_dir in subdirs
    assert old_dir not in subdirs


def test_filter_exif_data(dummy_processor):
    raw_files = [
        {"FileName": "test.NEF", "Model": "Nikon D850"},
        {"FileName": "test2.NEF"},  # 欠損あり
    ]

    dummy_processor.exif_fields = ["FileName", "Model"]
    filtered = dummy_processor.filter_exif_data(raw_files)

    assert filtered[0]["Model"] == "Nikon D850"
    assert filtered[1]["Model"] == "N/A"


def test_write_csv_creates_file(tmp_path, dummy_processor):
    data = [
        {"FileName": "test.NEF", "Model": "Nikon"},
        {"FileName": "test2.NEF", "Model": "Canon"},
    ]
    file_path = tmp_path / "output.csv"

    dummy_processor.exif_fields = ["FileName", "Model"]
    dummy_processor.write_csv(file_path, data)

    assert file_path.exists()
    with file_path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["Model"] == "Nikon"

@pytest.mark.skip(reason="process_directory() は現在未実装")
def test_process_directory_no_raw_files(dummy_processor, mocker):
    mocker.patch.object(dummy_processor.exif_handler, "raw_extensions", [".NEF"])
    mocker.patch.object(dummy_processor.exif_handler, "read_files", return_value=[])

    dummy_processor.process_directory(Path("/dummy/path"))

    dummy_processor.logger.warning.assert_called_once()


def test_write_csv_thread_safe(dummy_processor):
    file_path = dummy_processor.temp_dir / "thread_safe_test.csv"
    dummy_processor.exif_fields = ["FileName", "Model"]
    dummy_processor.append_mode = True
    num_threads = 5
    rows_per_thread = 10

    def write_task(start_index):
        data = [
            {"FileName": f"IMG_{i}.NEF", "Model": f"Camera-{i}"}
            for i in range(start_index, start_index + rows_per_thread)
        ]
        dummy_processor.write_csv(file_path, data)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(write_task, i * rows_per_thread) for i in range(num_threads)]
        for future in futures:
            future.result()

    # 検証
    assert file_path.exists()
    with file_path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == num_threads * rows_per_thread
        filenames = [row["FileName"] for row in reader]
        assert len(set(filenames)) == len(filenames)  # 重複なし

def test_get_data_returns_expected(monkeypatch, dummy_processor):
    # モック用のraw_filesデータ
    sample_raw_files = [
        {"SourceFile": "/path/to/file1.NEF", "Field1": "Value1"},
        {"SourceFile": "/path/to/file2.NEF", "Field1": "Value2"},
    ]

    # exif_handler.read_files をモックして上記リストを返すようにする
    monkeypatch.setattr(dummy_processor.exif_handler, "read_files", lambda path, **kwargs: sample_raw_files)
    # get_target_subdirectories もモックして一つのディレクトリだけ返す
    monkeypatch.setattr(dummy_processor, "get_target_subdirectories", lambda base_path: [Path("/path/to")])

    results = dummy_processor.get_data()

    assert len(results) == 2
    assert results[0]["path"] == "/path/to/file1.NEF"
    assert results[0]["directory"] == "/path/to"
    assert results[1]["exif_raw"]["Field1"] == "Value2"


def test_process_batch_calls_write_csv(monkeypatch, dummy_processor, tmp_path):
    called = []

    def mock_write_csv(self, file_path, data):
        print(f"[MOCK] write_csv called: {file_path}")
        print(f"[MOCK] data: {data}")
        called.append((file_path, data))

    dummy_processor.temp_dir = tmp_path
    dummy_processor.exif_fields = ["FileName", "Model"]  # Model も追加して完全性を保証

    monkeypatch.setattr(NEFFileBatchProcess, "write_csv", mock_write_csv)

    batch = [
        {"subdir_name": "sub1", "exif_raw": {"FileName": "a.NEF", "Model": "X1"}},
        {"subdir_name": "sub1", "exif_raw": {"FileName": "b.NEF", "Model": "X2"}},
        {"subdir_name": "sub2", "exif_raw": {"FileName": "c.NEF", "Model": "X3"}},
    ]

    dummy_processor._process_batch(batch)

    print(f"[DEBUG] called: {called}")
    assert len(called) == 2  # sub1, sub2
    assert all(len(data) > 0 for _, data in called)