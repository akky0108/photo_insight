# src/batch_processor/evaluation_rank/writer.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence


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


def collect_extra_columns(rows: List[Dict[str, Any]]) -> List[str]:
    """
    extra列は「存在するものだけ」末尾に追加（漏れ防止 & 安定）

    追加: 目・表情など運用判断に必要な派生列も拾う（例: eye_closed_prob_best）
    """
    extra_cols = set()

    # 追加したい “可変で増えていく列” のprefix群
    allow_prefixes = (
        "score_",
        "contrib_",
        "eye_",        # eye_closed_prob_best, eye_state, etc
        "debug_",      # debug_* 系
        "expr_",       # expression_* を付けるなら
    )

    # prefix じゃないけど欲しい固定列があればここに（将来用）
    allow_exact = {
        # "eye_closed_prob_best",
        # "eye_patch_size_best",
        # "eye_state",
    }

    for r in rows:
        for k in r.keys():
            if not isinstance(k, str):
                continue
            if k in allow_exact:
                extra_cols.add(k)
                continue
            if k.startswith(allow_prefixes):
                extra_cols.add(k)

    return sorted(extra_cols)


def build_columns(base_columns: Sequence[str], extra_columns: Sequence[str]) -> List[str]:
    """
    columns を確定する。
    - base → extra（昇順）→ tail（固定で末尾）
    - tail は必ず末尾に揃うように、前段から除外して最後に付与する
    """
    tail = [
        "lr_keywords",
        "lr_rating",
        "lr_color_label",
        "lr_labelcolor_key",
        "lr_label_display",
        "accepted_reason",
    ]

    tail_set = set(tail)

    # 末尾固定列は前段から除外（順序の崩れを防ぐ）
    base = [c for c in base_columns if c not in tail_set]
    extra = [c for c in extra_columns if c not in tail_set]

    cols = list(base) + list(extra) + tail

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
