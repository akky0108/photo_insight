# src/tools/brightness_comp_validate.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "1"

# =========================
# SSOT filenames (tests import these)
# =========================
SUMMARY_FILENAME = "brightness_comp_summary.json"
PAIRS_FILENAME = "brightness_comp_pairs.csv"

# =========================
# Rules (SSOT)
# =========================
DEFAULT_RULES: Dict[str, Any] = {
    "min_n": 30,
    "checks": {
        # 1) 補正後も序列が壊れてない（強め）
        "corr_before_vs_adj_min": 0.90,
        # 2) 変化量が極端じゃない（0..1スコアで 0.25=1段階）
        "p95_abs_delta_max": 0.25,
        # 3) 明るさに対して意味のある方向へ動いてる（例: 暗いほど救済＝負の相関）
        "corr_brightness_vs_delta_max": -0.20,
    },
    # 少数サンプルは skip にする（fail にしない）
    "skip_if_insufficient_n": True,
}

# =========================
# helpers
# =========================


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        if v in ("", None):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _safe_bool01(v: Any) -> int:
    if v is None or v == "":
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return 1 if int(v) != 0 else 0
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n"):
        return 0
    try:
        return 1 if int(float(s)) != 0 else 0
    except Exception:
        return 0


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _percentile(sorted_vals: Sequence[float], q01: float) -> float:
    """
    sorted_vals: 昇順
    q01: 0..1
    """
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_vals[0])

    q01 = max(0.0, min(1.0, float(q01)))
    pos = (n - 1) * q01
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """
    Pearson correlation with NaN/inf filtered out beforehand.
    Returns NaN if not computable.
    """
    n = min(len(xs), len(ys))
    if n <= 1:
        return float("nan")

    mx = sum(xs) / n
    my = sum(ys) / n
    num = 0.0
    dx2 = 0.0
    dy2 = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        num += dx * dy
        dx2 += dx * dx
        dy2 += dy * dy

    den = math.sqrt(dx2) * math.sqrt(dy2)
    if den <= 0.0:
        return float("nan")
    return num / den


def _filter_pairs(*cols: Sequence[float]) -> Tuple[List[float], ...]:
    """
    同じ index の値が全部 finite のものだけ残す。
    """
    n = min(len(c) for c in cols) if cols else 0
    out: List[List[float]] = [[] for _ in cols]
    for i in range(n):
        ok = True
        for c in cols:
            if not _is_finite(c[i]):
                ok = False
                break
        if not ok:
            continue
        for j, c in enumerate(cols):
            out[j].append(float(c[i]))
    return tuple(out)  # type: ignore


# =========================
# metrics spec
# =========================


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    score_col: str
    adjusted_col: str
    brightness_col: str = "mean_brightness"


DEFAULT_METRICS: List[MetricSpec] = [
    MetricSpec(
        metric="blurriness",
        score_col="blurriness_score",
        adjusted_col="blurriness_score_brightness_adjusted",
    ),
    MetricSpec(
        metric="noise",
        score_col="noise_score",
        adjusted_col="noise_score_brightness_adjusted",
    ),
    # face系（存在するCSVなら拾う）
    MetricSpec(
        metric="face_blurriness",
        score_col="face_blurriness_score",
        adjusted_col="face_blurriness_score_brightness_adjusted",
        brightness_col="face_mean_brightness",
    ),
]


def _required_cols(spec: MetricSpec) -> List[str]:
    return [spec.brightness_col, spec.score_col, spec.adjusted_col]


# =========================
# IO
# =========================


def _resolve_paths_by_glob(pattern: str) -> List[Path]:
    files = sorted(glob.glob(pattern, recursive=True))
    return [Path(x) for x in files if x and Path(x).is_file()]


def _read_csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        return [str(x).strip() for x in header if str(x).strip() != ""]
    except Exception:
        return []


def _iter_csv_rows(paths: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for p in paths:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if isinstance(row, dict):
                    row["_source_csv"] = str(p)
                    yield row


# =========================
# core compute
# =========================


def compute_checks_for_group(
    rows: List[Dict[str, Any]], spec: MetricSpec
) -> Dict[str, Any]:
    xs_brightness: List[float] = []
    ys_before: List[float] = []
    ys_adj: List[float] = []
    deltas: List[float] = []

    for r in rows:
        b = _safe_float(r.get(spec.brightness_col))
        before = _safe_float(r.get(spec.score_col))
        adj = _safe_float(r.get(spec.adjusted_col))
        if not (_is_finite(b) and _is_finite(before) and _is_finite(adj)):
            continue
        d = adj - before
        xs_brightness.append(float(b))
        ys_before.append(float(before))
        ys_adj.append(float(adj))
        deltas.append(float(d))

    xs_brightness, ys_before, ys_adj, deltas = _filter_pairs(
        xs_brightness, ys_before, ys_adj, deltas
    )

    n = len(deltas)
    if n == 0:
        return {
            "n": 0,
            "corr_before_vs_adj": float("nan"),
            "p95_abs_delta": float("nan"),
            "corr_brightness_vs_delta": float("nan"),
        }

    corr_before_vs_adj = _pearson_corr(ys_before, ys_adj)
    abs_d = sorted([abs(x) for x in deltas if _is_finite(x)])
    p95_abs_delta = _percentile(abs_d, 0.95) if abs_d else float("nan")
    corr_brightness_vs_delta = _pearson_corr(xs_brightness, deltas)

    return {
        "n": n,
        "corr_before_vs_adj": corr_before_vs_adj,
        "p95_abs_delta": p95_abs_delta,
        "corr_brightness_vs_delta": corr_brightness_vs_delta,
    }


def compute_pass_fail(
    *, metrics_summary: Dict[str, Any], rules: Dict[str, Any]
) -> Dict[str, Any]:
    """
    metrics_summary:
      {metric: {group: {n, corr_before_vs_adj, p95_abs_delta, corr_brightness_vs_delta}}}
    """
    checks_cfg = (rules.get("checks") or {}) if isinstance(rules, dict) else {}
    min_n = int(rules.get("min_n", 30))
    skip_if_insufficient = bool(rules.get("skip_if_insufficient_n", True))

    thr_corr_min = float(checks_cfg.get("corr_before_vs_adj_min", 0.90))
    thr_p95_max = float(checks_cfg.get("p95_abs_delta_max", 0.25))
    thr_corr_brightness_max = float(
        checks_cfg.get("corr_brightness_vs_delta_max", -0.20)
    )

    result = "pass"
    fail_reasons: List[str] = []
    checks_out: Dict[str, Any] = {}

    any_evaluated = False  # n>=min_n のチェックを1回でも実行したか

    for metric, per_group in metrics_summary.items():
        if not isinstance(per_group, dict):
            continue
        checks_out.setdefault(metric, {})
        for group, stats in per_group.items():
            if not isinstance(stats, dict):
                continue

            n = int(stats.get("n", 0) or 0)
            c1 = float(stats.get("corr_before_vs_adj", float("nan")))
            p95 = float(stats.get("p95_abs_delta", float("nan")))
            c2 = float(stats.get("corr_brightness_vs_delta", float("nan")))

            group_checks: Dict[str, Any] = {
                "n": n,
                "corr_before_vs_adj_min": {
                    "value": c1,
                    "threshold": thr_corr_min,
                    "pass": None,
                },
                "p95_abs_delta_max": {
                    "value": p95,
                    "threshold": thr_p95_max,
                    "pass": None,
                },
                "corr_brightness_vs_delta_max": {
                    "value": c2,
                    "threshold": thr_corr_brightness_max,
                    "pass": None,
                },
                "group_result": None,
                "group_fail_reasons": [],
                "group_skip_reason": "",
            }

            if n < min_n:
                if skip_if_insufficient:
                    group_checks["group_result"] = "skip"
                    group_checks["group_skip_reason"] = f"insufficient_n({n} < {min_n})"
                    checks_out[metric][group] = group_checks
                    continue
                # skipしない設定の場合は fail 寄りで続行
                group_checks["group_result"] = "fail"
                group_checks["group_fail_reasons"].append(
                    f"insufficient_n({n} < {min_n})"
                )

            # ここから評価開始
            any_evaluated = True

            ok1 = _is_finite(c1) and (c1 >= thr_corr_min)
            group_checks["corr_before_vs_adj_min"]["pass"] = bool(ok1)
            if not ok1:
                group_checks["group_fail_reasons"].append(
                    f"{metric}/{group}: corr_before_vs_adj={c1:.3f} < {thr_corr_min:.3f}"
                )

            ok2 = _is_finite(p95) and (p95 <= thr_p95_max)
            group_checks["p95_abs_delta_max"]["pass"] = bool(ok2)
            if not ok2:
                group_checks["group_fail_reasons"].append(
                    f"{metric}/{group}: p95_abs_delta={p95:.3f} > {thr_p95_max:.3f}"
                )

            ok3 = _is_finite(c2) and (c2 <= thr_corr_brightness_max)
            group_checks["corr_brightness_vs_delta_max"]["pass"] = bool(ok3)
            if not ok3:
                group_checks["group_fail_reasons"].append(
                    f"{metric}/{group}: corr_brightness_vs_delta={c2:.3f} > {thr_corr_brightness_max:.3f}"
                )

            if group_checks["group_fail_reasons"]:
                group_checks["group_result"] = "fail"
                result = "fail"
                fail_reasons.extend(group_checks["group_fail_reasons"])
            else:
                group_checks["group_result"] = "pass"

            checks_out[metric][group] = group_checks

    # 全部 skip しか無いなら “pass” にしない
    skip_reason = ""
    if (not any_evaluated) and result != "fail":
        result = "skip"
        skip_reason = "no_groups_evaluated_due_to_insufficient_n_or_no_valid_rows"

    return {
        "result": result,
        "skip_reason": skip_reason,
        "fail_reasons": fail_reasons,
        "checks": checks_out,
        "rules": {
            "min_n": min_n,
            "skip_if_insufficient_n": skip_if_insufficient,
            "checks": {
                "corr_before_vs_adj_min": thr_corr_min,
                "p95_abs_delta_max": thr_p95_max,
                "corr_brightness_vs_delta_max": thr_corr_brightness_max,
            },
        },
    }


# =========================
# discovery (compat / utility)
# =========================


def discover_ranking_files(*, root_dir: Path, date: Optional[str]) -> List[str]:
    """
    output 配下から evaluation_ranking_*.csv を探索する。

    date:
      None → 最新日付のファイルのみ返す
      "YYYY-MM-DD" → 指定日付のみ返す
    """
    root_dir = Path(root_dir)
    pattern = str(root_dir / "**" / "evaluation_ranking_*.csv")
    paths = _resolve_paths_by_glob(pattern)

    if not paths:
        return []

    if date:
        target = f"evaluation_ranking_{date}.csv"
        return [str(p) for p in paths if p.name == target]

    def _extract_date(p: Path) -> str:
        name = p.stem  # evaluation_ranking_YYYY-MM-DD
        parts = name.split("_")
        return parts[-1] if parts else ""

    paths_sorted = sorted(paths, key=lambda p: _extract_date(p), reverse=True)
    latest_date = _extract_date(paths_sorted[0])

    return [str(p) for p in paths_sorted if _extract_date(p) == latest_date]


# =========================
# output writers
# =========================


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_pairs_csv(path: Path, pairs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "file_name",
        "group",
        "metric",
        "brightness",
        "before",
        "adjusted",
        "delta",
        "_source_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in pairs:
            w.writerow(r)


# =========================
# public API (tests call this)
# =========================


def run(
    *,
    # --- new API ---
    root_dir: Optional[Path] = None,
    date: Optional[str] = None,
    # --- legacy API ---
    ranking_glob: Optional[str] = None,
    out_dir: Path,
    emit_pairs_csv: bool = False,
    rules: Optional[Dict[str, Any]] = None,
    metrics: Optional[List[MetricSpec]] = None,
    fail_on: bool = False,
) -> Path:
    """
    new API:
      run(root_dir=Path("output"), date=None|"YYYY-MM-DD", out_dir=Path("out"))

    legacy API:
      run(ranking_glob="output/**/evaluation_ranking_*.csv", out_dir=Path("out"))

    out_dir: output directory for SSOT artifacts
    emit_pairs_csv: if True, also emit PAIRS_FILENAME
    rules: override DEFAULT_RULES (merged shallowly)
    metrics: override DEFAULT_METRICS
    fail_on: if True, raise SystemExit(2) when result=="fail" (CI向け)

    return: summary_path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # rules merge (shallow)
    effective_rules = dict(DEFAULT_RULES)
    if isinstance(rules, dict):
        for k, v in rules.items():
            if k == "checks" and isinstance(v, dict):
                merged_checks = dict(effective_rules.get("checks") or {})
                merged_checks.update(v)
                effective_rules["checks"] = merged_checks
            else:
                effective_rules[k] = v

    specs = metrics if metrics is not None else list(DEFAULT_METRICS)

    # resolve paths
    if root_dir is not None:
        files = discover_ranking_files(root_dir=Path(root_dir), date=date)
        paths = [Path(p) for p in files]
        if ranking_glob is None:
            ranking_glob = str(Path(root_dir) / "**" / "evaluation_ranking_*.csv")
    else:
        if not ranking_glob:
            raise TypeError("run() requires either root_dir=... or ranking_glob=...")
        paths = _resolve_paths_by_glob(str(ranking_glob))

    # no file matched → emit skip summary with SSOT keys present
    if not paths:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "inputs": {
                "n_files": 0,
                "n_rows": 0,
                "n_rows_face": 0,
                "n_rows_non_face": 0,
                "ranking_glob": str(ranking_glob or ""),
                "root_dir": str(root_dir) if root_dir is not None else None,
                "date": date,
                # SSOT for diagnostics (always present)
                "files_used_per_metric": {},
                "files_skipped_per_metric": {},
                "skipped_files_reason": {},
                "n_rows_valid": {},
            },
            # 互換性MAX: top-levelにも同じキーを置く
            "files_used_per_metric": {},
            "files_skipped_per_metric": {},
            "skipped_files_reason": {},
            "n_rows_valid": {},
            "result": "skip",
            "skip_reason": f"no_files_matched: {ranking_glob}",
            "files": [],
            "metrics": {},
            "rules": effective_rules,
            "checks": {},
            "fail_reasons": [],
        }
        summary_path = out_dir / SUMMARY_FILENAME
        _write_json(summary_path, summary)
        return summary_path

    # build header map (for per-metric file filtering)
    headers_by_file: Dict[str, List[str]] = {str(p): _read_csv_header(p) for p in paths}
    header_sets: Dict[str, set] = {k: set(v) for k, v in headers_by_file.items()}

    # read all rows (for group row counts SSOT)
    all_rows_all_files = list(_iter_csv_rows(paths))

    def _group_name(r: Dict[str, Any]) -> str:
        fd = _safe_bool01(r.get("face_detected"))
        return "face" if fd == 1 else "non_face"

    groups_all_files: Dict[str, List[Dict[str, Any]]] = {
        "all": list(all_rows_all_files),
        "face": [],
        "non_face": [],
    }
    for r in all_rows_all_files:
        groups_all_files[_group_name(r)].append(r)

    # per metric: filter files by required columns
    files_used_per_metric: Dict[str, List[str]] = {}
    files_skipped_per_metric: Dict[str, List[str]] = {}
    skipped_files_reason: Dict[str, Dict[str, Any]] = {}
    n_rows_valid: Dict[str, Dict[str, int]] = {}

    metrics_summary: Dict[str, Any] = {}
    pairs_rows: List[Dict[str, Any]] = []

    for spec in specs:
        metric = spec.metric
        req = _required_cols(spec)

        used_files: List[str] = []
        skipped_files: List[str] = []
        reasons: Dict[str, Any] = {}

        for p in paths:
            ps = str(p)
            hs = header_sets.get(ps, set())
            missing = [c for c in req if c not in hs]
            if missing:
                skipped_files.append(ps)
                reasons[ps] = {"missing_columns": missing}
            else:
                used_files.append(ps)

        files_used_per_metric[metric] = used_files
        files_skipped_per_metric[metric] = skipped_files
        skipped_files_reason[metric] = reasons

        # rows for this metric: only from used_files
        used_set = set(used_files)
        rows_metric_all = [
            r for r in all_rows_all_files if str(r.get("_source_csv") or "") in used_set
        ]

        # group split (this metric only)
        groups_metric: Dict[str, List[Dict[str, Any]]] = {
            "all": [],
            "face": [],
            "non_face": [],
        }
        for r in rows_metric_all:
            g = _group_name(r)
            groups_metric["all"].append(r)
            groups_metric[g].append(r)

        # compute stats + valid row counts
        metrics_summary.setdefault(metric, {})
        n_rows_valid.setdefault(metric, {"all": 0, "face": 0, "non_face": 0})

        for gname, rows in groups_metric.items():
            stats = compute_checks_for_group(rows, spec)
            metrics_summary[metric][gname] = stats
            n_rows_valid[metric][gname] = int(stats.get("n", 0) or 0)

        if emit_pairs_csv:
            # pairs は all だけ吐く
            for r in groups_metric["all"]:
                b = _safe_float(r.get(spec.brightness_col))
                before = _safe_float(r.get(spec.score_col))
                adj = _safe_float(r.get(spec.adjusted_col))
                if not (_is_finite(b) and _is_finite(before) and _is_finite(adj)):
                    continue
                pairs_rows.append(
                    {
                        "file_name": str(r.get("file_name") or r.get("filename") or ""),
                        "group": (
                            "face"
                            if _safe_bool01(r.get("face_detected")) == 1
                            else "non_face"
                        ),
                        "metric": metric,
                        "brightness": b,
                        "before": before,
                        "adjusted": adj,
                        "delta": adj - before,
                        "_source_csv": str(r.get("_source_csv") or ""),
                    }
                )

    verdict = compute_pass_fail(metrics_summary=metrics_summary, rules=effective_rules)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "n_files": len(paths),
            "n_rows": len(all_rows_all_files),
            "n_rows_face": len(groups_all_files["face"]),
            "n_rows_non_face": len(groups_all_files["non_face"]),
            "ranking_glob": str(ranking_glob or ""),
            "root_dir": str(root_dir) if root_dir is not None else None,
            "date": date,
            # new SSOTs (diagnostics)
            "files_used_per_metric": files_used_per_metric,
            "files_skipped_per_metric": files_skipped_per_metric,
            "skipped_files_reason": skipped_files_reason,
            "n_rows_valid": n_rows_valid,
        },
        # 互換性MAX: top-levelにも同じキーを置く（tests/運用スクリプトがどっち見てもOK）
        "files_used_per_metric": files_used_per_metric,
        "files_skipped_per_metric": files_skipped_per_metric,
        "skipped_files_reason": skipped_files_reason,
        "n_rows_valid": n_rows_valid,
        "files": [str(p) for p in paths],
        "metrics": metrics_summary,
        "rules": verdict.get("rules"),
        "checks": verdict.get("checks"),
        "result": verdict.get("result"),
        "skip_reason": verdict.get("skip_reason"),
        "fail_reasons": verdict.get("fail_reasons"),
    }

    summary_path = out_dir / SUMMARY_FILENAME
    _write_json(summary_path, summary)

    if emit_pairs_csv:
        pairs_path = out_dir / PAIRS_FILENAME
        _write_pairs_csv(pairs_path, pairs_rows)

    if fail_on and summary.get("result") == "fail":
        raise SystemExit(2)

    return summary_path


# =========================
# CLI (legacy, keep simple)
# =========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate brightness-compensation behavior and emit SSOT artifacts."
    )
    parser.add_argument(
        "--ranking_glob",
        required=True,
        help='e.g. "output/**/evaluation_ranking_*.csv"',
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output dir for summary/pairs (SSOT).",
    )
    parser.add_argument(
        "--emit_pairs_csv",
        action="store_true",
        help=f"Also emit {PAIRS_FILENAME}.",
    )
    parser.add_argument(
        "--fail_on",
        action="store_true",
        help="Exit with code 2 if result == fail (for CI).",
    )
    parser.add_argument(
        "--rules_json",
        default=None,
        help="Optional JSON string to override DEFAULT_RULES shallowly.",
    )
    args = parser.parse_args()

    rules = None
    if args.rules_json:
        try:
            rules = json.loads(args.rules_json)
        except Exception:
            rules = None

    _ = run(
        ranking_glob=str(args.ranking_glob),
        out_dir=Path(args.out_dir),
        emit_pairs_csv=bool(args.emit_pairs_csv),
        rules=rules,
        fail_on=bool(args.fail_on),
    )


if __name__ == "__main__":
    main()
