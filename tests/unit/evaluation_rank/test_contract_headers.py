# tests/unit/evaluation_rank/test_contract_headers.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from photo_insight.batch_processor.evaluation_rank.contract import (
    INPUT_REQUIRED_COLUMNS,
    OUTPUT_COLUMNS,
)


def find_repo_root(start: Path) -> Path:
    """
    `.git` を辿ってリポジトリルートを見つける。
    見つからなければ start を返す（CI/特殊環境での安全策）。
    """
    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return start.resolve()


PROJECT_ROOT = find_repo_root(Path(__file__))


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            pytest.fail(f"CSV is empty: {path}")


def _latest_csv(dir_path: Path, pattern: str) -> Path | None:
    """
    例: temp/evaluation_results_*.csv のうち最新を返す。
    """
    if not dir_path.exists():
        return None
    files = sorted(dir_path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def test_input_contract_has_no_duplicates():
    assert len(INPUT_REQUIRED_COLUMNS) == len(set(INPUT_REQUIRED_COLUMNS)), "INPUT_REQUIRED_COLUMNS has duplicates"


def test_output_contract_has_no_duplicates():
    assert len(OUTPUT_COLUMNS) == len(set(OUTPUT_COLUMNS)), "OUTPUT_COLUMNS has duplicates"


def test_output_contract_includes_accepted_reason_once():
    assert OUTPUT_COLUMNS.count("accepted_reason") == 1


@pytest.mark.parametrize("col", ["file_name", "overall_score", "accepted_flag", "secondary_accept_flag"])
def test_output_contract_has_core_columns(col: str):
    assert col in OUTPUT_COLUMNS


def test_contract_output_order_is_fixed_snapshot():
    """
    列順が変わるとLightroom運用や後段分析が壊れるので、ここで固定する。
    """
    assert OUTPUT_COLUMNS.index("overall_score") < OUTPUT_COLUMNS.index("lr_keywords")
    assert OUTPUT_COLUMNS.index("lr_keywords") < OUTPUT_COLUMNS.index("accepted_reason")


def test_sample_csv_headers_match_contract_if_present():
    """
    ローカルにサンプルCSVが存在する場合だけ、ヘッダを契約と突き合わせる。
    - CI環境にCSVが無いケースを想定して skip にする
    - 日付固定ではなく、最新の1件を拾う
    """
    input_csv = _latest_csv(PROJECT_ROOT / "temp", "evaluation_results_*.csv")
    output_csv = _latest_csv(PROJECT_ROOT / "output", "evaluation_ranking_*.csv")

    if input_csv is None and output_csv is None:
        pytest.skip("No sample CSVs found under temp/ or output/")

    if input_csv is not None:
        hdr = _read_csv_header(input_csv)
        missing = [c for c in INPUT_REQUIRED_COLUMNS if c not in hdr]
        assert not missing, f"Input CSV missing required columns: {missing}"

    if output_csv is not None:
        hdr = _read_csv_header(output_csv)
        assert hdr == OUTPUT_COLUMNS, "Output CSV header/order must match OUTPUT_COLUMNS exactly"
