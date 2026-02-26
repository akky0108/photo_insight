from __future__ import annotations

import csv
from pathlib import Path


def test_write_csv_contract_writes_header_even_if_rows_empty(tmp_path: Path):
    from photo_insight.batch_processor.evaluation_rank.writer import write_csv_contract
    from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS

    out = tmp_path / "out.csv"
    write_csv_contract(out, rows=[], columns=OUTPUT_COLUMNS)

    with out.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == list(OUTPUT_COLUMNS)


def test_write_ranking_csv_enforces_output_columns_and_order(tmp_path: Path):
    from photo_insight.batch_processor.evaluation_rank.writer import write_ranking_csv
    from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS

    out = tmp_path / "evaluation_ranking_2099-01-01.csv"
    rows = [{"file_name": "IMG_0001.NEF", "overall_score": 0.1}]

    cols = write_ranking_csv(output_csv=out, rows=rows, sort_for_ranking=False)
    assert cols == list(OUTPUT_COLUMNS)

    with out.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == list(OUTPUT_COLUMNS)
