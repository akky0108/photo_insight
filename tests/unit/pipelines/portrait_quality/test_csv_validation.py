import csv

from photo_insight.pipelines.portrait_quality.portrait_quality_batch_processor import (
    PortraitQualityBatchProcessor,
)


def _make_processor(tmp_path, image_csv_file, base_directory):
    processor = PortraitQualityBatchProcessor(auto_load=False)
    processor.image_csv_file = str(image_csv_file)
    processor.base_directory = str(base_directory)
    processor.invalid_rows_csv_file = str(tmp_path / "invalid_rows.csv")
    return processor


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_load_image_data_returns_empty_when_required_columns_are_missing(tmp_path):
    base_dir = tmp_path / "images"
    base_dir.mkdir()

    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        fieldnames=["FileName", "Orientation"],
        rows=[
            {"FileName": "a.jpg", "Orientation": "1"},
        ],
    )

    processor = _make_processor(tmp_path, csv_path, base_dir)

    rows = processor.load_image_data()

    assert rows == []
    assert not (tmp_path / "invalid_rows.csv").exists()


def test_load_image_data_skips_duplicate_file_name_and_writes_invalid_rows(tmp_path):
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    (base_dir / "a.jpg").write_bytes(b"dummy")

    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        fieldnames=["FileName", "Orientation", "BitDepth"],
        rows=[
            {"FileName": "a.jpg", "Orientation": "1", "BitDepth": "14"},
            {"FileName": "a.jpg", "Orientation": "1", "BitDepth": "14"},
        ],
    )

    processor = _make_processor(tmp_path, csv_path, base_dir)

    rows = processor.load_image_data()
    invalid_rows = _read_csv(tmp_path / "invalid_rows.csv")

    assert len(rows) == 1
    assert rows[0]["file_name"] == "a.jpg"
    assert len(invalid_rows) == 1
    assert invalid_rows[0]["file_name"] == "a.jpg"
    assert invalid_rows[0]["reason"] == "duplicate_file_name"


def test_load_image_data_skips_invalid_orientation_and_writes_invalid_rows(tmp_path):
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    (base_dir / "a.jpg").write_bytes(b"dummy")

    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        fieldnames=["FileName", "Orientation", "BitDepth"],
        rows=[
            {"FileName": "a.jpg", "Orientation": "not-int", "BitDepth": "14"},
        ],
    )

    processor = _make_processor(tmp_path, csv_path, base_dir)

    rows = processor.load_image_data()
    invalid_rows = _read_csv(tmp_path / "invalid_rows.csv")

    assert rows == []
    assert len(invalid_rows) == 1
    assert invalid_rows[0]["file_name"] == "a.jpg"
    assert invalid_rows[0]["orientation"] == "not-int"
    assert invalid_rows[0]["reason"] == "invalid_orientation"


def test_load_image_data_skips_invalid_bit_depth_and_writes_invalid_rows(tmp_path):
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    (base_dir / "a.jpg").write_bytes(b"dummy")

    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        fieldnames=["FileName", "Orientation", "BitDepth"],
        rows=[
            {"FileName": "a.jpg", "Orientation": "1", "BitDepth": "not-int"},
        ],
    )

    processor = _make_processor(tmp_path, csv_path, base_dir)

    rows = processor.load_image_data()
    invalid_rows = _read_csv(tmp_path / "invalid_rows.csv")

    assert rows == []
    assert len(invalid_rows) == 1
    assert invalid_rows[0]["file_name"] == "a.jpg"
    assert invalid_rows[0]["bit_depth"] == "not-int"
    assert invalid_rows[0]["reason"] == "invalid_bit_depth"


def test_load_image_data_skips_missing_image_file_and_writes_invalid_rows(tmp_path):
    base_dir = tmp_path / "images"
    base_dir.mkdir()

    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        fieldnames=["FileName", "Orientation", "BitDepth"],
        rows=[
            {"FileName": "missing.jpg", "Orientation": "1", "BitDepth": "14"},
        ],
    )

    processor = _make_processor(tmp_path, csv_path, base_dir)

    rows = processor.load_image_data()
    invalid_rows = _read_csv(tmp_path / "invalid_rows.csv")

    assert rows == []
    assert len(invalid_rows) == 1
    assert invalid_rows[0]["file_name"] == "missing.jpg"
    assert invalid_rows[0]["reason"] == "file_not_found"


def test_load_image_data_writes_only_invalid_rows_when_valid_and_invalid_rows_are_mixed(tmp_path):
    base_dir = tmp_path / "images"
    base_dir.mkdir()
    (base_dir / "valid.jpg").write_bytes(b"dummy")
    (base_dir / "invalid.jpg").write_bytes(b"dummy")

    csv_path = tmp_path / "input.csv"
    _write_csv(
        csv_path,
        fieldnames=["FileName", "Orientation", "BitDepth"],
        rows=[
            {"FileName": "valid.jpg", "Orientation": "1", "BitDepth": "14"},
            {"FileName": "invalid.jpg", "Orientation": "bad", "BitDepth": "14"},
        ],
    )

    processor = _make_processor(tmp_path, csv_path, base_dir)

    rows = processor.load_image_data()
    invalid_rows = _read_csv(tmp_path / "invalid_rows.csv")

    assert len(rows) == 1
    assert rows[0]["file_name"] == "valid.jpg"

    assert len(invalid_rows) == 1
    assert invalid_rows[0]["file_name"] == "invalid.jpg"
    assert invalid_rows[0]["reason"] == "invalid_orientation"
