# tests/unit/evaluation_rank/test_writer.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import pytest

from batch_processor.evaluation_rank.contract import (
    INPUT_REQUIRED_COLUMNS,
    OUTPUT_COLUMNS,
    validate_input_contract,
)
from batch_processor.evaluation_rank.writer import (
    safe_int_flag,
    sort_rows_for_ranking,
    write_ranking_csv,    
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def _mk_min_row(**overrides: Any) -> Dict[str, Any]:
    """
    OUTPUT_COLUMNS を埋める必要はない。
    writer が欠損列を "" 埋めするので、必要最低限だけ入れる。
    """
    base: Dict[str, Any] = {
        "file_name": "a.NEF",
        "category": "portrait",
        "overall_score": 80.0,
        "flag": 0,
        "accepted_flag": 0,
        "secondary_accept_flag": 0,
    }
    base.update(overrides)
    return base


# =========================
# safe_int_flag
# =========================

@pytest.mark.parametrize(
    "value, expected",
    [
        (None, 0),
        ("", 0),
        (False, 0),
        (True, 1),
        (0, 0),
        (1, 1),
        (2, 1),
        (0.0, 0),
        (1.0, 1),
        ("0", 0),
        ("1", 1),
        ("true", 1),
        ("TRUE", 1),
        ("false", 0),
        ("False", 0),
        ("yes", 1),
        ("no", 0),
        (" 1 ", 1),
        (" 0 ", 0),
        ("0.0", 0),
        ("2.0", 1),
        ("garbage", 0),
    ],
)
def test_safe_int_flag_normalizes(value: Any, expected: int) -> None:
    assert safe_int_flag(value) == expected


# =========================
# write_ranking_csv contract
# =========================

def test_write_ranking_csv_writes_exact_contract_columns(tmp_path: Path) -> None:
    out = tmp_path / "ranking.csv"
    rows = [
        _mk_min_row(file_name="x.NEF", accepted_flag=1, overall_score=88.0),
        _mk_min_row(file_name="y.NEF", secondary_accept_flag=1, overall_score=77.0),
    ]

    cols = write_ranking_csv(output_csv=out, rows=rows, sort_for_ranking=False)

    # return value must be contract
    assert cols == list(OUTPUT_COLUMNS)

    header, body = _read_csv(out)
    assert header == list(OUTPUT_COLUMNS)
    assert len(body) == 2

    # all contract columns exist in every row
    for r in body:
        assert set(r.keys()) == set(OUTPUT_COLUMNS)


def test_write_ranking_csv_fills_missing_columns_with_empty_string(tmp_path: Path) -> None:
    out = tmp_path / "ranking.csv"
    rows = [
        # overall_score だけ入れて他は欠損でも落ちない
        {"file_name": "a.NEF", "overall_score": 50.0},
    ]

    write_ranking_csv(output_csv=out, rows=rows, sort_for_ranking=False)
    header, body = _read_csv(out)

    assert header == list(OUTPUT_COLUMNS)
    assert len(body) == 1

    r0 = body[0]
    # 欠損していた列は "" 埋め
    assert r0["category"] == ""
    assert r0["shot_type"] == ""
    assert r0["lr_keywords"] == ""
    # 入れていた値は残る（CSVなので文字列）
    assert r0["file_name"] == "a.NEF"
    assert r0["overall_score"] in ("50.0", "50")  # writerは数値加工しないので str 化のみ


def test_write_ranking_csv_normalizes_flags_to_01(tmp_path: Path) -> None:
    out = tmp_path / "ranking.csv"
    rows = [
        _mk_min_row(file_name="a.NEF", accepted_flag="TRUE", secondary_accept_flag="False", flag="1"),
        _mk_min_row(file_name="b.NEF", accepted_flag="0", secondary_accept_flag="1", flag="no"),
    ]

    write_ranking_csv(output_csv=out, rows=rows, sort_for_ranking=False)
    _, body = _read_csv(out)

    assert body[0]["accepted_flag"] == "1"
    assert body[0]["secondary_accept_flag"] == "0"
    assert body[0]["flag"] == "1"

    assert body[1]["accepted_flag"] == "0"
    assert body[1]["secondary_accept_flag"] == "1"
    assert body[1]["flag"] == "0"


def test_write_ranking_csv_accepts_filename_alias(tmp_path: Path) -> None:
    out = tmp_path / "ranking.csv"

    # file_name が無いが filename がある古いデータ
    rows = [
        {"filename": "legacy.NEF", "overall_score": 60.0, "category": "portrait"},
    ]

    write_ranking_csv(output_csv=out, rows=rows, sort_for_ranking=False)
    _, body = _read_csv(out)

    assert body[0]["file_name"] == "legacy.NEF"


def test_write_ranking_csv_ignores_extras_keys(tmp_path: Path) -> None:
    out = tmp_path / "ranking.csv"
    rows = [
        _mk_min_row(
            file_name="a.NEF",
            overall_score=70.0,
            faces_parse_reason="faces_parse:ok:json",  # extra
            main_subject_center_invalid_reason="center_calc_failed:face_box",  # extra
        )
    ]

    write_ranking_csv(output_csv=out, rows=rows, sort_for_ranking=False)
    header, body = _read_csv(out)

    assert header == list(OUTPUT_COLUMNS)
    assert len(body) == 1
    # extras は出力されない（契約が守られる）
    assert "faces_parse_reason" not in body[0]
    assert "main_subject_center_invalid_reason" not in body[0]


# =========================
# sort_rows_for_ranking
# =========================

def test_sort_rows_for_ranking_orders_as_spec() -> None:
    rows = [
        _mk_min_row(file_name="c.NEF", category="non_face", accepted_flag=1, overall_score=99.0),
        _mk_min_row(file_name="b.NEF", category="portrait", accepted_flag=0, secondary_accept_flag=1, overall_score=80.0),
        _mk_min_row(file_name="a.NEF", category="portrait", accepted_flag=1, overall_score=70.0),
        _mk_min_row(file_name="d.NEF", category="portrait", accepted_flag=0, flag=1, overall_score=95.0),
        _mk_min_row(file_name="e.NEF", category="portrait", accepted_flag=0, overall_score=96.0),
    ]

    sorted_rows = sort_rows_for_ranking(rows)

    # 優先順位:
    # 1) portrait -> non_face
    # 2) accepted_flag desc
    # 3) secondary desc
    # 4) flag desc
    # 5) overall desc
    # 6) fname asc
    assert [r["file_name"] for r in sorted_rows] == [
        "a.NEF",  # portrait + accepted
        "b.NEF",  # portrait + secondary
        "d.NEF",  # portrait + flag
        "e.NEF",  # portrait + none (overall high)
        "c.NEF",  # non_face (even though accepted & overall high, category is after)
    ]


def test_write_ranking_csv_sort_for_ranking_true_applies_sort(tmp_path: Path) -> None:
    out = tmp_path / "ranking.csv"
    rows = [
        _mk_min_row(file_name="b.NEF", category="portrait", accepted_flag=0, overall_score=99.0),
        _mk_min_row(file_name="a.NEF", category="portrait", accepted_flag=1, overall_score=10.0),
    ]

    write_ranking_csv(output_csv=out, rows=rows, sort_for_ranking=True)
    _, body = _read_csv(out)

    # accepted が先に来る
    assert body[0]["file_name"] == "a.NEF"
    assert body[0]["accepted_flag"] == "1"


# =========================
# input contract validation
# =========================

def test_validate_input_contract_ok_with_required_columns() -> None:
    """
    INPUT_REQUIRED_COLUMNS が揃っていれば例外にならない。
    """
    # NOTE: import path はあなたの実ファイルに合わせて調整してね
    # 例: from batch_processor.evaluation_rank.evaluation_rank_batch_processor import _validate_input_contract
    from batch_processor.evaluation_rank.contract import INPUT_REQUIRED_COLUMNS

    header = list(INPUT_REQUIRED_COLUMNS)
    validate_input_contract(header=header, csv_path=Path("dummy.csv"))


def test_validate_input_contract_raises_with_missing_columns_message() -> None:
    """
    1列でも欠けていたら ValueError になり、missing数と一部列名がメッセージに入る。
    """
    from batch_processor.evaluation_rank.contract import INPUT_REQUIRED_COLUMNS

    # わざと2つ欠けさせる
    header = list(INPUT_REQUIRED_COLUMNS)
    missing_cols = ["file_name", "faces"]
    header = [c for c in header if c not in set(missing_cols)]

    with pytest.raises(ValueError) as e:
        validate_input_contract(header=header, csv_path=Path("in.csv"))

    msg = str(e.value)
    assert "Input CSV contract violation" in msg
    assert "missing 2 columns" in msg
    # 欠けた列名がメッセージに出る
    assert "file_name" in msg
    assert "faces" in msg


def test_validate_input_contract_message_truncates_after_20_columns() -> None:
    """
    missing が 20 を超えると preview が省略表記される。
    """
    from batch_processor.evaluation_rank.contract import INPUT_REQUIRED_COLUMNS

    # required のうち、先頭1個だけ残して大量に欠けさせる
    header = [INPUT_REQUIRED_COLUMNS[0]]

    with pytest.raises(ValueError) as e:
        validate_input_contract(header=header, csv_path=Path("in.csv"))

    msg = str(e.value)
    # 20超 missing なので suffix が付く想定
    assert "...(+" in msg
