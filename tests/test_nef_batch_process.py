import csv
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor

from batch_processor.nef_batch_process import NEFFileBatchProcess
import batch_framework.utils.io_utils as io_utils

# --- 共通フィクスチャ ---
@pytest.fixture
def dummy_processor(tmp_path):
    processor = NEFFileBatchProcess(config_path=None)
    processor.project_root = tmp_path
    processor.temp_dir = tmp_path / "temp"
    processor.temp_dir.mkdir(parents=True, exist_ok=True)
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

def test_process_directory_no_raw_files(dummy_processor, mocker, tmp_path):
    mocker.patch.object(dummy_processor.exif_handler, "raw_extensions", [".NEF"])
    mocker.patch.object(dummy_processor.exif_handler, "read_files", return_value=[])

    dummy_processor.temp_dir = tmp_path / "temp"
    dummy_processor.temp_dir.mkdir(parents=True, exist_ok=True)

    dummy_processor.process_directory("sample_dir", [])

    dummy_processor.logger.warning.assert_any_call("[単体処理] 対象ファイルなし: sample_dir")

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

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(write_task, i * rows_per_thread) for i in range(num_threads)]
        for future in futures:
            future.result()

    # 検証：重複しないファイル名と行数を確認（ヘッダ重複を除外）
    assert file_path.exists()
    with file_path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        rows = [row for row in reader if row["FileName"] != "FileName"]  # ヘッダ行が混入していた場合除外
        assert len(rows) == num_threads * rows_per_thread
        filenames = [row["FileName"] for row in rows]
        assert len(set(filenames)) == len(filenames)  # 重複なし

def test_get_data_with_target_dir(monkeypatch, dummy_processor):
    sample_path = Path("/mock/target_dir")
    sample_raw_files = [
        {"SourceFile": "/mock/target_dir/file1.NEF", "Model": "Z9"},
        {"SourceFile": "/mock/target_dir/file2.NEF", "Model": "Z8"},
    ]
    monkeypatch.setattr(dummy_processor.exif_handler, "read_files", lambda p, **kwargs: sample_raw_files)
    results = dummy_processor.get_data(target_dir=sample_path)

    assert len(results) == 2
    assert all(result["subdir_name"] == sample_path.name for result in results)
    assert results[0]["filename"] == "file1.NEF"

def test_get_data_default_scans_all(monkeypatch, dummy_processor):
    dir1 = Path("/mock/dir1")
    dir2 = Path("/mock/dir2")
    monkeypatch.setattr(dummy_processor, "get_target_subdirectories", lambda base_path: [dir1, dir2])

    def mock_read_files(path, **kwargs):
        if path == str(dir1):
            return [{"SourceFile": "/mock/dir1/a.NEF", "ISO": "100"}]
        if path == str(dir2):
            return [{"SourceFile": "/mock/dir2/b.NEF", "ISO": "200"}]
        return []

    monkeypatch.setattr(dummy_processor.exif_handler, "read_files", mock_read_files)

    results = dummy_processor.get_data()

    assert len(results) == 2
    subdirs = [r["subdir_name"] for r in results]
    assert "dir1" in subdirs and "dir2" in subdirs

def test_get_data_returns_expected(monkeypatch, dummy_processor):
    sample_raw_files = [
        {"SourceFile": "/path/to/file1.NEF", "Field1": "Value1"},
        {"SourceFile": "/path/to/file2.NEF", "Field1": "Value2"},
    ]
    monkeypatch.setattr(dummy_processor.exif_handler, "read_files", lambda path, **kwargs: sample_raw_files)
    monkeypatch.setattr(dummy_processor, "get_target_subdirectories", lambda base_path: [Path("/path/to")])

    results = dummy_processor.get_data()

    assert len(results) == 2
    assert results[0]["path"] == "/path/to/file1.NEF"
    assert results[0]["directory"] == "/path/to"
    assert results[1]["exif_raw"]["Field1"] == "Value2"

def test_process_batch_calls_write_csv(monkeypatch, dummy_processor, tmp_path):
    called = []

    def mock_write_csv_with_lock(file_path, data, fieldnames, lock, append, logger):
        called.append((str(file_path), data))

    dummy_processor.temp_dir = tmp_path
    dummy_processor.exif_fields = ["FileName", "Model"]
    monkeypatch.setattr("batch_processor.nef_batch_process.write_csv_with_lock", mock_write_csv_with_lock)

    batch = [
        {"subdir_name": "sub1", "exif_raw": {"FileName": "a.NEF", "Model": "X1"}},
        {"subdir_name": "sub1", "exif_raw": {"FileName": "b.NEF", "Model": "X2"}},
        {"subdir_name": "sub2", "exif_raw": {"FileName": "c.NEF", "Model": "X3"}},
    ]

    dummy_processor._process_batch(batch)

    assert len(called) == 2
    called_files = [Path(fp).name for fp, _ in called]
    assert "sub1_raw_exif_data.csv" in called_files
    assert "sub2_raw_exif_data.csv" in called_files
    assert all(len(data) > 0 for _, data in called)

def test_execute_with_target_dir(monkeypatch, dummy_processor, tmp_path):
    # 対象ディレクトリ（仮）
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    # get_data → バッチ処理対象の mock データを返す
    monkeypatch.setattr(dummy_processor, "get_data", lambda target_dir=None: [
        {"subdir_name": "target", "exif_raw": {"FileName": "a.NEF", "Model": "X1"}}
    ])

    # process の呼び出し確認用
    called = {}
    monkeypatch.setattr(dummy_processor, "process", lambda data: called.update({"called": data}))

    # 実行
    dummy_processor.execute(target_dir=target_dir)

    # 検証
    assert "called" in called
    assert len(called["called"]) == 1
    assert called["called"][0]["subdir_name"] == "target"