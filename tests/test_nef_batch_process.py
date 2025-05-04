import csv
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from batch_processor.nef_batch_process import NEFFileBatchProcess  # ← あなたのモジュールパスに合わせてね

@pytest.fixture
def dummy_processor(tmp_path):
    """NEFFileBatchProcessのダミーインスタンス"""
    processor = NEFFileBatchProcess(config_path=None)
    processor.project_root = tmp_path  # 仮のプロジェクトルート
    processor.temp_dir = tmp_path / "temp"
    processor.temp_dir.mkdir()
    processor.logger = MagicMock()
    return processor

def test_get_target_subdirectories(tmp_path, dummy_processor):
    """ターゲット日付以降のサブディレクトリだけ取得できるか"""
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_file = old_dir / "dummy.txt"
    old_file.write_text("dummy")  # ダミーファイルを作る

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_file = new_dir / "dummy.txt"
    new_file.write_text("dummy")  # ダミーファイルを作る

    # old_dirと中のファイルのタイムスタンプを1年前にする
    import os, time
    old_timestamp = time.time() - (60 * 60 * 24 * 365)  # 1年前
    os.utime(old_dir, (old_timestamp, old_timestamp))
    os.utime(old_file, (old_timestamp, old_timestamp))

    dummy_processor.target_date = datetime.now() - timedelta(minutes=1)  # 1分前

    subdirs = dummy_processor.get_target_subdirectories(tmp_path)

    assert new_dir in subdirs
    assert old_dir not in subdirs

def test_filter_exif_data(dummy_processor):
    """EXIFフィールド抽出時に欠損データをN/Aに埋めるか"""
    raw_files = [
        {"FileName": "test.NEF", "Model": "Nikon D850"},
        {"FileName": "test2.NEF"}  # Modelフィールド欠損
    ]

    dummy_processor.exif_fields = ["FileName", "Model"]
    filtered = dummy_processor.filter_exif_data(raw_files)

    assert filtered[0]["Model"] == "Nikon D850"
    assert filtered[1]["Model"] == "N/A"

def test_write_csv_creates_file(tmp_path, dummy_processor):
    """CSVファイルが正しく書き込まれるか"""
    data = [
        {"FileName": "test.NEF", "Model": "Nikon"},
        {"FileName": "test2.NEF", "Model": "Canon"}
    ]
    file_path = tmp_path / "output.csv"

    dummy_processor.exif_fields = ["FileName", "Model"]
    dummy_processor.write_csv(file_path, data)

    assert file_path.exists()
    with file_path.open(newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["Model"] == "Nikon"

def test_process_directory_no_raw_files(dummy_processor, mocker):
    """RAWファイルがないとき警告が出てスキップされるか"""
    mocker.patch.object(dummy_processor.exif_handler, 'raw_extensions', ['.NEF'])
    mocker.patch.object(dummy_processor.exif_handler, 'read_files', return_value=[])

    dummy_processor.process_directory(Path("/dummy/path"))

    dummy_processor.logger.warning.assert_called_once()
