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
    """
    base + extra + tail の順に結合（tailは本処理で固定）
    """
    tail = [
        "lr_keywords",
        "lr_rating",
        "lr_color_label",
        "lr_labelcolor_key",
        "lr_label_display",
        "accepted_reason",
    ]
    return list(base_columns) + list(extra_columns) + tail


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in columns})


def write_ranking_csv(
    *,
    output_csv: Path,
    rows: List[Dict[str, Any]],
    base_columns: Sequence[str],
) -> List[str]:
    """
    ranking出力専用:
    - extra columns を集める
    - columns を確定する
    - CSV を書き出す

    return: 実際に書いた columns（テストやログに使える）
    """
    extra_columns = collect_extra_columns(rows)
    columns = build_columns(base_columns, extra_columns)
    write_csv(output_csv, rows, columns)
    return columns
