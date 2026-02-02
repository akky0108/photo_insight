#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_score_distributions.py

目的:
- A/B 2つのCSVに対して、score列の妥当性チェックを通った metric のみを使い
  score分布の比較CSV/PNGを生成する。

A確定仕様:
- score妥当性チェック:
    - 値域が概ね 0..1（微小許容）
    - 離散率 >= 95%（{0,0.25,0.5,0.75,1.0}）
    - *_raw 併存必須（例: contrast_score には contrast_raw が同一CSVに必要）
    - 満たさない metric はスキップ＋警告
- 対象メトリクス（固定）:
    sharpness/blurriness/contrast/noise/face_sharpness/face_blurriness/face_contrast

出力:
- out_dir/comparison_scores_summary.csv
- out_dir/png/<metric>_score.png  (A/Bのratio棒グラフ)

使い方:
  python tools/compare_score_distributions.py --csv-a ... --csv-b ... --out-dir ...
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


DISCRETE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
DISCRETE_SET = set(DISCRETE_LEVELS)

TARGET_SCORE_COLS = [
    "sharpness_score",
    "blurriness_score",
    "contrast_score",
    "noise_score",
    "face_sharpness_score",
    "face_blurriness_score",
    "face_contrast_score",
]

TECH_TARGET_RANGES = {
    1.0: (0.05, 0.10),
    0.75: (0.20, 0.30),
    0.5: (0.25, 0.35),
    0.25: (0.15, 0.25),
    0.0: (0.10, 0.15),
}
TECH_TARGET_CENTER = {lv: (lo + hi) / 2.0 for lv, (lo, hi) in TECH_TARGET_RANGES.items()}


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if math.isnan(v):
            return None
        return v
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        v = float(s)
        if math.isnan(v):
            return None
        return v
    except ValueError:
        return None


def round_to_quarter(x: float) -> float:
    return round(x * 4.0) / 4.0


def read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    return rows, headers


def collect_column_values(rows: List[Dict[str, str]], col: str) -> List[float]:
    vals: List[float] = []
    for r in rows:
        v = safe_float(r.get(col))
        if v is None:
            continue
        vals.append(v)
    return vals


def require_raw_column(headers: List[str], score_col: str) -> Optional[str]:
    if not score_col.endswith("_score"):
        return None
    raw_col = score_col.replace("_score", "_raw")
    return raw_col if raw_col in headers else None


@dataclass
class ScoreValidity:
    ok: bool
    n: int
    in_range_ratio: float
    discrete_ratio: float
    min_v: float
    max_v: float
    reason: str


def validate_score_series(
    values: List[float],
    *,
    min_samples: int,
    min_discrete_ratio: float,
    range_min: float = -0.001,
    range_max: float = 1.001,
) -> ScoreValidity:
    if not values:
        return ScoreValidity(False, 0, 0.0, 0.0, float("nan"), float("nan"), "no numeric values")

    n = len(values)
    min_v = min(values)
    max_v = max(values)

    in_range = [v for v in values if (range_min <= v <= range_max)]
    in_range_ratio = len(in_range) / n if n else 0.0

    if n < min_samples:
        return ScoreValidity(False, n, in_range_ratio, 0.0, min_v, max_v, f"too few samples (n={n} < {min_samples})")

    # score列ならほぼ 0..1 に収まるはず。外れが多いなら誤認可能性高い。
    if in_range_ratio < 0.98:
        return ScoreValidity(False, n, in_range_ratio, 0.0, min_v, max_v, f"in-range ratio too low ({in_range_ratio:.3f} < 0.98)")

    rounded = [round_to_quarter(v) for v in in_range]
    discrete = [v for v in rounded if v in DISCRETE_SET]
    discrete_ratio = len(discrete) / len(in_range) if in_range else 0.0

    if discrete_ratio < min_discrete_ratio:
        return ScoreValidity(False, n, in_range_ratio, discrete_ratio, min_v, max_v,
                             f"discrete ratio too low ({discrete_ratio:.3f} < {min_discrete_ratio})")

    return ScoreValidity(True, n, in_range_ratio, discrete_ratio, min_v, max_v, "ok")


def distribution_counts(values: List[float]) -> Dict[float, int]:
    counts = {lv: 0 for lv in DISCRETE_LEVELS}
    for v in values:
        if not (-0.001 <= v <= 1.001):
            continue
        rv = round_to_quarter(v)
        if rv in counts:
            counts[rv] += 1
    return counts


def counts_to_ratios(counts: Dict[float, int]) -> Dict[float, float]:
    total = sum(counts.values())
    if total == 0:
        return {lv: 0.0 for lv in DISCRETE_LEVELS}
    return {lv: counts.get(lv, 0) / total for lv in DISCRETE_LEVELS}


def saturation_ratio(ratios: Dict[float, float]) -> float:
    return ratios.get(0.0, 0.0) + ratios.get(1.0, 0.0)


def tech_target_flags(ratios: Dict[float, float]) -> Dict[float, str]:
    flags: Dict[float, str] = {}
    for lv, (lo, hi) in TECH_TARGET_RANGES.items():
        v = ratios.get(lv, 0.0)
        flags[lv] = "in" if (lo <= v <= hi) else ("low" if v < lo else "high")
    return flags


def tech_target_l1(ratios: Dict[float, float]) -> float:
    return sum(abs(ratios.get(lv, 0.0) - TECH_TARGET_CENTER[lv]) for lv in DISCRETE_LEVELS)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_distribution_bar(
    metric_label: str,
    ratios_a: Dict[float, float],
    ratios_b: Dict[float, float],
    label_a: str,
    label_b: str,
    out_png: Path,
) -> None:
    x = list(range(len(DISCRETE_LEVELS)))
    y_a = [ratios_a.get(lv, 0.0) for lv in DISCRETE_LEVELS]
    y_b = [ratios_b.get(lv, 0.0) for lv in DISCRETE_LEVELS]

    plt.figure(figsize=(8, 4.2))
    width = 0.38
    plt.bar([i - width / 2 for i in x], y_a, width=width, label=label_a)
    plt.bar([i + width / 2 for i in x], y_b, width=width, label=label_b)
    plt.xticks(x, [str(lv) for lv in DISCRETE_LEVELS])
    plt.title(f"{metric_label}: score distribution (ratio)")
    plt.xlabel("score")
    plt.ylabel("ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-a", required=True, help="baseline CSV (before)")
    ap.add_argument("--csv-b", required=True, help="candidate CSV (after)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out-dir", default="output/score_dist_compare")

    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--min-discrete-ratio", type=float, default=0.95)

    ap.add_argument("--face-saturation-warn", type=float, default=0.80,
                    help="face系で(0+1)飽和率がこの値を超えたら警告（ただしスキップしない）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "png")

    rows_a, headers_a = read_csv_rows(Path(args.csv_a))
    rows_b, headers_b = read_csv_rows(Path(args.csv_b))

    if not rows_a:
        raise RuntimeError("csv-a has no rows")
    if not rows_b:
        raise RuntimeError("csv-b has no rows")

    common_headers = set(headers_a).intersection(set(headers_b))
    target_cols = [c for c in TARGET_SCORE_COLS if c in common_headers]

    if not target_cols:
        warn("No target score columns found in common headers.")
        return 2

    info(f"Targets (common): {target_cols}")

    summary_rows: List[Dict[str, object]] = []
    valid_cols: List[str] = []

    for score_col in target_cols:
        raw_a = require_raw_column(headers_a, score_col)
        raw_b = require_raw_column(headers_b, score_col)
        if raw_a is None or raw_b is None:
            warn(
                f"Skip '{score_col}': required raw column missing.\n"
                f"  expected: '{score_col.replace('_score','_raw')}' | A has? {'yes' if raw_a else 'no'} | B has? {'yes' if raw_b else 'no'}"
            )
            continue

        vals_a = collect_column_values(rows_a, score_col)
        vals_b = collect_column_values(rows_b, score_col)

        v_a = validate_score_series(vals_a, min_samples=args.min_samples, min_discrete_ratio=args.min_discrete_ratio)
        v_b = validate_score_series(vals_b, min_samples=args.min_samples, min_discrete_ratio=args.min_discrete_ratio)

        if not v_a.ok or not v_b.ok:
            warn(
                f"Skip '{score_col}': invalid score series.\n"
                f"  A: ok={v_a.ok} n={v_a.n} range=[{v_a.min_v:.3g},{v_a.max_v:.3g}] in_range={v_a.in_range_ratio:.3f} discrete={v_a.discrete_ratio:.3f} reason={v_a.reason}\n"
                f"  B: ok={v_b.ok} n={v_b.n} range=[{v_b.min_v:.3g},{v_b.max_v:.3g}] in_range={v_b.in_range_ratio:.3f} discrete={v_b.discrete_ratio:.3f} reason={v_b.reason}"
            )
            continue

        valid_cols.append(score_col)

        counts_a = distribution_counts(vals_a)
        counts_b = distribution_counts(vals_b)
        ratios_a = counts_to_ratios(counts_a)
        ratios_b = counts_to_ratios(counts_b)

        sat_a = saturation_ratio(ratios_a)
        sat_b = saturation_ratio(ratios_b)

        is_face = score_col.startswith("face_")
        is_tech = score_col in {"sharpness_score", "blurriness_score", "contrast_score", "noise_score"}

        if is_face and (sat_a >= args.face_saturation_warn or sat_b >= args.face_saturation_warn):
            warn(f"Face metric '{score_col}' extreme? saturation(0+1): A={sat_a:.3f}, B={sat_b:.3f} (warn>= {args.face_saturation_warn})")

        row: Dict[str, object] = {
            "metric": score_col,
            "type": "face" if is_face else ("tech" if is_tech else "other"),
            "raw_required": "yes",
            "n_a": v_a.n,
            "n_b": v_b.n,
            "in_range_ratio_a": f"{v_a.in_range_ratio:.4f}",
            "in_range_ratio_b": f"{v_b.in_range_ratio:.4f}",
            "discrete_ratio_a": f"{v_a.discrete_ratio:.4f}",
            "discrete_ratio_b": f"{v_b.discrete_ratio:.4f}",
            "saturation_0plus1_a": f"{sat_a:.4f}",
            "saturation_0plus1_b": f"{sat_b:.4f}",
            "tech_target_l1_a": f"{tech_target_l1(ratios_a):.6f}" if is_tech else "",
            "tech_target_l1_b": f"{tech_target_l1(ratios_b):.6f}" if is_tech else "",
        }

        flags_a = tech_target_flags(ratios_a) if is_tech else {lv: "" for lv in DISCRETE_LEVELS}
        flags_b = tech_target_flags(ratios_b) if is_tech else {lv: "" for lv in DISCRETE_LEVELS}

        for lv in DISCRETE_LEVELS:
            row[f"ratio_a_{lv}"] = f"{ratios_a.get(lv, 0.0):.4f}"
            row[f"ratio_b_{lv}"] = f"{ratios_b.get(lv, 0.0):.4f}"
            row[f"tech_target_flag_a_{lv}"] = flags_a[lv] if is_tech else ""
            row[f"tech_target_flag_b_{lv}"] = flags_b[lv] if is_tech else ""

        summary_rows.append(row)

        out_png = out_dir / "png" / f"{score_col}_score.png"
        plot_distribution_bar(
            metric_label=score_col.replace("_score", ""),
            ratios_a=ratios_a,
            ratios_b=ratios_b,
            label_a=args.label_a,
            label_b=args.label_b,
            out_png=out_png,
        )

    if not summary_rows:
        warn("No valid metrics found after checks.")
        return 3

    out_csv = out_dir / "comparison_scores_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    info(f"Saved summary CSV: {out_csv}")
    info(f"Saved PNG dir    : {out_dir / 'png'}")
    info(f"Valid metrics    : {valid_cols}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
