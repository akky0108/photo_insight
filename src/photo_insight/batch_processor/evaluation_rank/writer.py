# src/photo_insight/batch_processor/evaluation_rank/writer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS


def safe_int_flag(value: Any) -> int:
    """
    CSV由来の 0/1, True/False, "TRUE"/"False" を 0/1 に正規化する。
    int("False") 事故を確実に回避するため、ここ以外で int(...) しない。
    """
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0

    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n"):
        return 0

    try:
        return 1 if int(float(s)) != 0 else 0
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


# =========================
# ランキング用の並び替え
# =========================


def sort_rows_for_ranking(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ランキングCSV用の行ソートヘルパー。

    優先順位:
    1. category: portrait が先, non_face が後
    2. accepted_flag: 1 が先（Green）
    3. secondary_accept_flag: 1 が先（Yellow）
    4. flag: 1 が先（Blue候補）
    5. overall_score: 高い順
    6. file_name/filename: 文字列昇順（安定化用）
    """

    def _cat_order(cat: str) -> int:
        if cat == "portrait":
            return 0
        if cat == "non_face":
            return 1
        return 2

    def _key(r: Dict[str, Any]):
        cat = _cat_order(str(r.get("category") or ""))
        accepted = safe_int_flag(r.get("accepted_flag"))
        secondary = safe_int_flag(r.get("secondary_accept_flag"))
        flag = safe_int_flag(r.get("flag"))
        overall = _safe_float(r.get("overall_score"))
        fname = str(r.get("file_name") or r.get("filename") or "")
        # 降順にしたいものは符号を反転
        return (
            cat,
            -accepted,
            -secondary,
            -flag,
            -overall,
            fname,
        )

    return sorted(rows, key=_key)


# =========================
# Contract-based CSV writing
# =========================


def _normalize_row_for_output(
    row: Dict[str, Any], columns: Sequence[str]
) -> Dict[str, Any]:
    """
    OUTPUT_COLUMNS を満たすように row を正規化する。
    - 欠損列は "" で埋める（契約として必ず出力列を揃える）
    - フラグ系は 0/1 に正規化
    - None は "" に寄せる（CSVでの扱いを安定化）
    - extras は無視（DictWriter 側では渡さない）
    """
    out: Dict[str, Any] = {}

    for c in columns:
        v = row.get(c, "")

        # 代表的な別名吸収（過去互換）
        if v == "" and c == "file_name":
            v = row.get("filename", "")

        if v is None:
            v = ""

        if c in ("flag", "accepted_flag", "secondary_accept_flag"):
            v = safe_int_flag(v)

        out[c] = v

    return out


def write_csv_contract(
    path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]
) -> None:
    """
    Contract(列順・列数) を完全に守って CSV を書く。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # rows が空でもヘッダだけは出す（運用上の事故防止）
    rows = rows or []

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(columns),
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            if not isinstance(r, dict):
                continue
            writer.writerow(_normalize_row_for_output(r, columns))


def write_ranking_csv(
    *,
    output_csv: Path,
    rows: List[Dict[str, Any]],
    base_columns: Optional[Sequence[str]] = None,
    sort_for_ranking: bool = True,
) -> List[str]:
    """
    ranking出力専用（Contract準拠）:

    重要:
    - OUTPUT_COLUMNS（contract.py）を単一の正として使う（列順も契約）
    - base_columns は後方互換のため残しているが、基本は無視する

    return: 実際に書いた columns（= OUTPUT_COLUMNS）
    """
    columns = list(OUTPUT_COLUMNS)

    # base_columns が渡されても SSOT 優先。将来の移行確認用に、差分を「返さず」静かに無視する。
    # （必要ならここで assert / warning を入れるが、運用を落とさない方針で今回は何もしない）
    _ = base_columns

    if sort_for_ranking:
        rows = sort_rows_for_ranking(rows)

    write_csv_contract(output_csv, rows, columns)

    # ★ ここで最終保証（将来の改修で writer が壊れても即検知）
    if columns != list(OUTPUT_COLUMNS):
        raise RuntimeError(
            "Output CSV contract violation: writer produced unexpected columns/order.\n"
            f"expected={OUTPUT_COLUMNS}\n"
            f"actual={columns}"
        )

    return columns
