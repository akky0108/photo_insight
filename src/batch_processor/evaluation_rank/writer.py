# src/batch_processor/evaluation_rank/writer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence


def collect_extra_columns(rows: List[Dict[str, Any]]) -> List[str]:
    """
    breakdown列は「存在するものだけ」末尾に追加（漏れ防止 & 安定）
    """
    extra_cols = set()
    for r in rows:
        for k in r.keys():
            if k.startswith("score_") or k.startswith("contrib_"):
                extra_cols.add(k)
    return sorted(extra_cols)


def build_columns(base_columns: Sequence[str], extra_columns: Sequence[str]) -> List[str]:
    tail = [
        "lr_keywords",
        "lr_rating",
        "lr_color_label",
        "lr_labelcolor_key",
        "lr_label_display",
        "accepted_reason",
    ]

    cols = list(base_columns) + list(extra_columns) + tail

    # 順序を保って重複削除
    seen = set()
    uniq: List[str] = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq



def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for r in rows:
            out = {}
            for k in columns:
                v = r.get(k)
                out[k] = "" if v is None else v
            writer.writerow(out)



# =========================
# ランキング用の並び替え
# =========================

def _safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def sort_rows_for_ranking(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ランキングCSV用の行ソートヘルパー。

    優先順位:
    1. category: portrait が先, non_face が後
    2. accepted_flag: 1 が先
    3. flag: 1 が先
    4. overall_score: 高い順
    5. file_name/filename: 文字列昇順（安定化用）
    """
    def _cat_order(cat: str) -> int:
        if cat == "portrait":
            return 0
        if cat == "non_face":
            return 1
        return 2  # 未設定などは最後

    def _key(r: Dict[str, Any]):
        cat = _cat_order(str(r.get("category") or ""))
        accepted = int(r.get("accepted_flag") or 0)
        flag = int(r.get("flag") or 0)
        overall = _safe_float(r.get("overall_score") or 0.0)
        fname = str(r.get("file_name") or r.get("filename") or "")
        # 降順にしたいものは符号を反転
        return (
            cat,
            -accepted,
            -flag,
            -overall,
            fname,
        )

    return sorted(rows, key=_key)


def write_ranking_csv(
    *,
    output_csv: Path,
    rows: List[Dict[str, Any]],
    base_columns: Sequence[str],
    sort_for_ranking: bool = True,
) -> List[str]:
    """
    ranking出力専用:
    - extra columns を集める
    - columns を確定する
    - （必要なら）行をランキング順にソートする
    - CSV を書き出す

    return: 実際に書いた columns（テストやログに使える）
    """
    if sort_for_ranking:
        rows = sort_rows_for_ranking(rows)

    extra_columns = collect_extra_columns(rows)
    columns = build_columns(base_columns, extra_columns)
    write_csv(output_csv, rows, columns)
    return columns
