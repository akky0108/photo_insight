# src/tools/brightness_comp_validate.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import glob as _glob


# =========================
# SSOT: Output contract
# =========================
SCHEMA_VERSION = "1.2"

SUMMARY_FILENAME = "brightness_validation_summary.json"

# ★ 既存テスト互換: pairs CSV 名を export
PAIRS_FILENAME = "brightness_validation_pairs.csv"


# =========================
# SSOT: file discovery
# =========================
RANKING_PATTERN = "evaluation_ranking_*.csv"
DATE_RE = re.compile(r"evaluation_ranking_(\d{4}-\d{2}-\d{2})\.csv")


# =========================
# SSOT: metrics
# =========================
# NOTE:
# - face_blurriness は face_mean_brightness を優先し、無い/不正なら mean_brightness へフォールバック
# - face_noise は「adjusted列」が現状パイプラインに無さそうなので今回は未登録（将来追加）
METRICS: Dict[str, Dict[str, Any]] = {
    "blurriness": {
        "score_col": "blurriness_score",
        "score_adj_col": "blurriness_score_brightness_adjusted",
        "raw_col": "blurriness_raw",
        "brightness_cols": ["mean_brightness"],
    },
    "noise": {
        "score_col": "noise_score",
        "score_adj_col": "noise_score_brightness_adjusted",
        "raw_col": "noise_raw",
        "brightness_cols": ["mean_brightness"],
    },
    "face_blurriness": {
        "score_col": "face_blurriness_score",
        "score_adj_col": "face_blurriness_score_brightness_adjusted",
        "raw_col": "face_blurriness_raw",
        "brightness_cols": ["face_mean_brightness", "mean_brightness"],  # ★fallback
    },
}


# =========================
# utils
# =========================

def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(x: Any) -> Optional[float]:
    try:
        if x in ("", None):
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _to_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x in ("", None):
        return False
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return False


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _extract_date_from_name(name: str) -> Optional[str]:
    m = DATE_RE.match(name)
    return m.group(1) if m else None


def discover_ranking_files(*, root_dir: Path, date: Optional[str]) -> List[str]:
    """
    SSOT:
    - date 指定あり → evaluation_ranking_{date}.csv のみ（最大1件）
    - date なし → 最新日付の 1件のみ（最大1件）
    """
    if not root_dir.exists():
        return []

    all_files = sorted(root_dir.rglob(RANKING_PATTERN))
    if not all_files:
        return []

    if date:
        target_name = f"evaluation_ranking_{date}.csv"
        matches = [str(p) for p in all_files if p.name == target_name]
        return matches[:1]

    dated: List[Tuple[str, Path]] = []
    for p in all_files:
        d = _extract_date_from_name(p.name)
        if d:
            dated.append((d, p))
    if not dated:
        return []

    dated.sort(key=lambda x: x[0], reverse=True)
    return [str(dated[0][1])]


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pick_brightness(row: Dict[str, Any], brightness_cols: List[str]) -> Optional[float]:
    for col in brightness_cols:
        v = _to_float(row.get(col))
        if v is not None:
            return v
    return None


def _metric_summary(
    *,
    rows: List[Dict[str, Any]],
    score_col: str,
    score_adj_col: str,
    raw_col: str,
    brightness_cols: List[str],
) -> Dict[str, Any]:
    xs, ys, rs, bs, deltas = [], [], [], [], []

    for r in rows:
        s = _to_float(r.get(score_col))
        a = _to_float(r.get(score_adj_col))
        raw = _to_float(r.get(raw_col))
        b = _pick_brightness(r, brightness_cols)

        if None in (s, a, raw, b):
            continue

        xs.append(s)
        ys.append(a)
        rs.append(raw)
        bs.append(b)
        deltas.append(a - s)

    abs_d = sorted(abs(d) for d in deltas)

    return {
        "score_col": score_col,
        "score_adj_col": score_adj_col,
        "raw_col": raw_col,
        "brightness_cols": list(brightness_cols),
        "n": len(xs),
        "corr_score_vs_adj": _pearson(xs, ys),
        "corr_raw_vs_adj": _pearson(rs, ys),
        "corr_brightness_vs_delta": _pearson(bs, deltas),
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "median_delta": (sorted(deltas)[len(deltas)//2] if deltas else 0.0),
        "p95_abs_delta": (abs_d[int(0.95 * len(abs_d))] if abs_d else 0.0),
    }


def _split_rows(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    all / face / non_face に分割
    """
    face = []
    non_face = []
    for r in rows:
        if _to_bool(r.get("face_detected")):
            face.append(r)
        else:
            non_face.append(r)
    return {"all": rows, "face": face, "non_face": non_face}


def _write_pairs_csv(
    *,
    out_path: Path,
    rows: List[Dict[str, Any]],
) -> None:
    """
    ★ 既存テスト互換: emit_pairs_csv=True のときに出す。
    各行×metric の “score→adjusted の差” を行で出力。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "metric",
        "file_name",
        "face_detected",
        "brightness",
        "score",
        "adjusted_score",
        "delta",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        for r in rows:
            fname = str(r.get("file_name") or r.get("filename") or "")
            face_detected = 1 if _to_bool(r.get("face_detected")) else 0

            for mname, spec in METRICS.items():
                s = _to_float(r.get(spec["score_col"]))
                a = _to_float(r.get(spec["score_adj_col"]))
                b = _pick_brightness(r, spec["brightness_cols"])
                if None in (s, a, b):
                    continue

                w.writerow(
                    {
                        "metric": mname,
                        "file_name": fname,
                        "face_detected": face_detected,
                        "brightness": b,
                        "score": s,
                        "adjusted_score": a,
                        "delta": a - s,
                    }
                )


# =========================
# public API (backward-compatible)
# =========================

def run(
    *,
    # --- new style ---
    root_dir: Optional[Path] = None,
    date: Optional[str] = None,
    # --- legacy style (tests/unit/tools/test_brightness_compensation_verify_tool.py 互換) ---
    ranking_glob: Optional[str] = None,
    # --- outputs ---
    out_dir: Path,
    emit_pairs_csv: bool = False,
) -> Path:
    """
    後方互換を維持した run()

    - 新: run(root_dir=Path("output"), date="YYYY-MM-DD", out_dir=Path(...))
    - 旧: run(ranking_glob="output/**/evaluation_ranking_*.csv", out_dir=..., emit_pairs_csv=True)
    """
    files: List[str] = []

    if ranking_glob:
        files = sorted(_glob.glob(ranking_glob, recursive=True))
    else:
        rd = root_dir or Path("output")
        files = discover_ranking_files(root_dir=rd, date=date)

    rows: List[Dict[str, Any]] = []
    for f in files:
        p = Path(f)
        if p.exists():
            rows.extend(_read_rows(p))

    groups = _split_rows(rows)

    metrics_out: Dict[str, Any] = {}
    for name, spec in METRICS.items():
        per_group: Dict[str, Any] = {}
        for gname, grows in groups.items():
            per_group[gname] = _metric_summary(rows=grows, **spec)
        metrics_out[name] = per_group

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso_utc(),
        "inputs": {
            "root_dir": str(root_dir) if root_dir is not None else None,
            "date": date,
            "ranking_glob": ranking_glob,
            "ranking_files": files,
            "n_files": len(files),
            "n_rows": len(rows),
            "n_rows_face": len(groups["face"]),
            "n_rows_non_face": len(groups["non_face"]),
        },
        "metrics": metrics_out,
        "notes": {
            "groups": ["all", "face", "non_face"],
            "face_blurriness_brightness_fallback": "face_mean_brightness -> mean_brightness",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if emit_pairs_csv:
        pairs_path = out_dir / PAIRS_FILENAME
        _write_pairs_csv(out_path=pairs_path, rows=rows)

    return summary_path


# =========================
# CLI
# =========================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", default="output")
    p.add_argument("--date", default=None)
    p.add_argument("--ranking-glob", default=None)
    p.add_argument("--out-dir", default="output/brightness_validation")
    p.add_argument("--emit-pairs-csv", action="store_true")
    args = p.parse_args()

    run(
        root_dir=Path(args.root_dir) if args.ranking_glob is None else None,
        date=args.date,
        ranking_glob=args.ranking_glob,
        out_dir=Path(args.out_dir),
        emit_pairs_csv=bool(args.emit_pairs_csv),
    )


if __name__ == "__main__":
    main()
