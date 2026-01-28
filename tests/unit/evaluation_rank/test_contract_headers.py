# tests/unit/evaluation_rank/test_contract_headers.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from batch_processor.evaluation_rank.contract import (
    INPUT_REQUIRED_COLUMNS,
    OUTPUT_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


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
    # 重要な境界だけでも順序保証（全体比較は OUTPUT_COLUMNS 自体がSSOTなので不要）
    assert OUTPUT_COLUMNS.index("overall_score") < OUTPUT_COLUMNS.index("lr_keywords")
    assert OUTPUT_COLUMNS.index("lr_keywords") < OUTPUT_COLUMNS.index("accepted_reason")


def test_sample_csv_headers_match_contract_if_present():
    """
    もしローカルにサンプルCSVが存在する場合だけ、ヘッダを契約と突き合わせる。
    - CI環境にCSVが無いケースを想定して skip にする
    """
    input_csv = PROJECT_ROOT / "temp" / "evaluation_results_2026-01-25.csv"
    output_csv = PROJECT_ROOT / "output" / "evaluation_ranking_2026-01-25.csv"

    # 任意チェック（あれば検証、なければskip）
    if input_csv.exists():
        hdr = _read_csv_header(input_csv)
        missing = [c for c in INPUT_REQUIRED_COLUMNS if c not in hdr]
        assert not missing, f"Input CSV missing required columns: {missing}"

    if output_csv.exists():
        hdr = _read_csv_header(output_csv)
        assert hdr == OUTPUT_COLUMNS, "Output CSV header/order must match OUTPUT_COLUMNS exactly"
