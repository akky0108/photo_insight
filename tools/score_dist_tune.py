#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_dist_tune.py

「現状の分布」→「目標分布に寄せる新閾値」→「raw→score再計算」→「分布比較」を一気に出す。

要件:
- temp/evaluation_results_*.csv をそのまま食わせて処理（デフォルトglob）
- 直近(更新時刻)のNファイルのみ対象（デフォルト5）
- *_raw と *_score を使用（対象は固定）
- score妥当性チェック導入（A確定）:
    - 値域 0..1（微小許容）
    - 離散率 >= 95%（{0,0.25,0.5,0.75,1.0}）
    - *_raw 併存必須
    - 満たさない metric はスキップ＋警告

対象メトリクス（固定）:
- 技術系: sharpness, blurriness, contrast, noise
- face系: face_sharpness, face_blurriness, face_contrast
※ "local_*" や body_composition は対象外（誤認混入を避ける）

目標分布（技術系）:
- 1.0: 5-10%, 0.75: 20-30%, 0.5: 25-35%, 0.25: 15-25%, 0.0: 10-15%
→ 自動閾値提案は「目標中心」に合わせた分位点境界を使う:
  0.0=12.5%, 0.25=20%, 0.5=30%, 0.75=25%, 1.0=7.5%
  境界: t0=12.5%, t1=32.5%, t2=62.5%, t3=87.5%

face系:
- 現段階は「極端寄り抑制」を“監視”に留める（警告と可視化）。
- 自動閾値提案の適用は --apply-face-auto で切替可能（デフォルトOFF推奨）。

出力:
- out_dir/new_params_used.json        (採用した閾値)
- out_dir/score_distribution_long.csv (long形式 current/new)
- out_dir/score_distribution_wide.csv (wide形式 + delta)
- out_dir/metric_summary.csv          (metricごとの要約：飽和率・目標差など)
- out_dir/plots/*_raw_hist.png
- out_dir/plots/*_score_compare.png

使い方:
  python tools/score_dist_tune.py
  python tools/score_dist_tune.py --n-files 5
  python tools/score_dist_tune.py --params-json temp/new_params.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")   # ★ headless環境対応

import matplotlib.pyplot as plt



DISCRETE_SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]
DISCRETE_SET = set(DISCRETE_SCORES)

# 対象固定（score_col -> raw_col を自動派生）
TARGET_SCORE_COLS = [
    "sharpness_score",
    "blurriness_score",
    "contrast_score",
    "noise_score",
    "face_sharpness_score",
    "face_blurriness_score",
    "face_contrast_score",
]

TECH_SCORE_COLS = {"sharpness_score", "blurriness_score", "contrast_score", "noise_score"}
FACE_SCORE_COLS = {"face_sharpness_score", "face_blurriness_score", "face_contrast_score"}

# 技術系の目標分布レンジ（割合）
TECH_TARGET_RANGES = {
    1.0: (0.05, 0.10),
    0.75: (0.20, 0.30),
    0.5: (0.25, 0.35),
    0.25: (0.15, 0.25),
    0.0: (0.10, 0.15),
}
TECH_TARGET_CENTER = {lv: (lo + hi) / 2.0 for lv, (lo, hi) in TECH_TARGET_RANGES.items()}

# 目標中心に合わせた分位点境界（rawが高いほど良い想定）
# t0: 12.5%, t1: 32.5%, t2: 62.5%, t3: 87.5%
TECH_TARGET_QUANTILES = [0.125, 0.325, 0.625, 0.875]


@dataclass
class Thresholds:
    """
    5段階離散スコア用の閾値（rawが高いほど良い想定）
      raw < t0       -> 0.0
      t0 <= raw < t1 -> 0.25
      t1 <= raw < t2 -> 0.5
      t2 <= raw < t3 -> 0.75
      t3 <= raw      -> 1.0
    """
    t0: float
    t1: float
    t2: float
    t3: float

    def to_list(self) -> List[float]:
        return [self.t0, self.t1, self.t2, self.t3]

    @staticmethod
    def from_list(xs: List[float]) -> "Thresholds":
        if len(xs) != 4:
            raise ValueError(f"thresholds must have 4 values, got {len(xs)}")
        return Thresholds(float(xs[0]), float(xs[1]), float(xs[2]), float(xs[3]))


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def find_latest_files(glob_pattern: str, n_files: int) -> List[Path]:
    paths = [Path(p) for p in sorted(Path().glob(glob_pattern))]
    if not paths:
        base = Path(glob_pattern).parent
        pat = Path(glob_pattern).name
        if base.exists():
            paths = list(base.glob(pat))

    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {glob_pattern}")

    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:n_files]
    return paths


def read_concat_csv(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file__"] = f.name
            dfs.append(df)
        except Exception as e:
            warn(f"failed to read: {f} ({e})")
    if not dfs:
        raise RuntimeError("All CSV reads failed.")
    return pd.concat(dfs, ignore_index=True)


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def round_to_quarter(x: float) -> float:
    return round(x * 4.0) / 4.0


def score_from_raw(raw: float, thr: Thresholds) -> float:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return float("nan")
    if raw < thr.t0:
        return 0.0
    if raw < thr.t1:
        return 0.25
    if raw < thr.t2:
        return 0.5
    if raw < thr.t3:
        return 0.75
    return 1.0


def compute_score_ratios(scores: pd.Series) -> Dict[float, float]:
    s = coerce_numeric(scores).dropna()
    if s.empty:
        return {v: 0.0 for v in DISCRETE_SCORES}

    # 0.25刻みに丸めて離散判定
    s = s.map(lambda x: round_to_quarter(float(x)))
    vc = s.value_counts()
    total = int(vc.sum())
    ratios = {v: float(vc.get(v, 0)) / total for v in DISCRETE_SCORES}
    return ratios


def compute_score_counts(scores: pd.Series) -> Dict[float, int]:
    s = coerce_numeric(scores).dropna()
    if s.empty:
        return {v: 0 for v in DISCRETE_SCORES}

    s = s.map(lambda x: round_to_quarter(float(x)))
    vc = s.value_counts()
    return {v: int(vc.get(v, 0)) for v in DISCRETE_SCORES}


def saturation_ratio(ratios: Dict[float, float]) -> float:
    return ratios.get(0.0, 0.0) + ratios.get(1.0, 0.0)


def discrete_ratio(scores: pd.Series) -> float:
    s = coerce_numeric(scores).dropna()
    if s.empty:
        return 0.0
    # 0..1 範囲を前提に in-range を取ってから離散率判定
    s = s[(s >= -0.001) & (s <= 1.001)]
    if s.empty:
        return 0.0
    r = s.map(lambda x: round_to_quarter(float(x)))
    ok = r.isin(list(DISCRETE_SET))
    return float(ok.mean())


def in_range_ratio(scores: pd.Series) -> float:
    s = coerce_numeric(scores).dropna()
    if s.empty:
        return 0.0
    ok = (s >= -0.001) & (s <= 1.001)
    return float(ok.mean())

def resolve_raw_col(df: pd.DataFrame, score_col: str) -> Tuple[Optional[str], str]:
    """
    score_col から raw_col を解決する。
    戻り値: (raw_col or None, note)
    """
    metric = score_col.replace("_score", "")

    # 第一候補: xxx_raw
    c1 = f"{metric}_raw"
    if c1 in df.columns:
        return c1, "raw"

    # noise は sigma_used を raw として扱える（noise_raw が無くても動かす）
    if metric == "noise" and "noise_sigma_used" in df.columns:
        return "noise_sigma_used", "fallback:sigma_used"

    if metric == "face_noise" and "face_noise_sigma_used" in df.columns:
        return "face_noise_sigma_used", "fallback:sigma_used"

    return None, "missing"


def validate_score_column(
    df: pd.DataFrame,
    score_col: str,
    raw_col: str,
    *,
    min_samples: int,
    min_discrete_ratio: float,
) -> Tuple[bool, str]:
    # raw併存必須
    if raw_col not in df.columns:
        return False, f"missing required raw column: {raw_col}"
    if score_col not in df.columns:
        return False, f"missing score column: {score_col}"

    s = coerce_numeric(df[score_col]).dropna()
    n = int(s.shape[0])
    if n < min_samples:
        return False, f"too few samples (n={n} < {min_samples})"

    r_in = in_range_ratio(df[score_col])
    if r_in < 0.98:
        return False, f"in-range ratio too low ({r_in:.3f} < 0.98); suspicious column"

    r_disc = discrete_ratio(df[score_col])
    if r_disc < min_discrete_ratio:
        return False, f"discrete ratio too low ({r_disc:.3f} < {min_discrete_ratio}); suspicious column"

    return True, "ok"


def propose_thresholds_for_target_distribution(raw_values: pd.Series) -> Thresholds:
    r = coerce_numeric(raw_values).dropna()
    if len(r) < 20:
        # データが少ない場合は等間隔でフォールバック（落ちないための保険）
        mn, mx = (float(r.min()), float(r.max())) if len(r) else (0.0, 1.0)
        step = (mx - mn) / 5.0 if mx > mn else 1.0
        return Thresholds(mn + step, mn + 2 * step, mn + 3 * step, mn + 4 * step)

    qs = r.quantile(TECH_TARGET_QUANTILES).tolist()
    return Thresholds.from_list([float(x) for x in qs])


def tech_target_flags(ratios: Dict[float, float]) -> Dict[float, str]:
    flags: Dict[float, str] = {}
    for lv, (lo, hi) in TECH_TARGET_RANGES.items():
        v = ratios.get(lv, 0.0)
        flags[lv] = "in" if (lo <= v <= hi) else ("low" if v < lo else "high")
    return flags


def tech_target_l1(ratios: Dict[float, float]) -> float:
    return sum(abs(ratios.get(lv, 0.0) - TECH_TARGET_CENTER[lv]) for lv in DISCRETE_SCORES)


def make_plots(
    out_dir: Path,
    metric: str,
    raw: pd.Series,
    cur_score: pd.Series,
    new_score: pd.Series,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_n = coerce_numeric(raw).dropna()
    cur_n = coerce_numeric(cur_score).dropna().map(lambda x: round_to_quarter(float(x)))
    new_n = coerce_numeric(new_score).dropna().map(lambda x: round_to_quarter(float(x)))

    # raw histogram
    plt.figure(figsize=(8, 4.2))
    if not raw_n.empty:
        plt.hist(raw_n, bins=50)
    plt.title(f"{metric}: raw distribution")
    plt.xlabel("raw")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_raw_hist.png", dpi=150)
    plt.close()

    def counts_for(series: pd.Series) -> List[int]:
        vc = series.value_counts()
        return [int(vc.get(v, 0)) for v in DISCRETE_SCORES]

    cur_counts = counts_for(cur_n)
    new_counts = counts_for(new_n)

    x = list(range(len(DISCRETE_SCORES)))
    width = 0.4
    plt.figure(figsize=(8, 4.2))
    plt.bar([i - width / 2 for i in x], cur_counts, width=width, label="current")
    plt.bar([i + width / 2 for i in x], new_counts, width=width, label="new")
    plt.title(f"{metric}: score distribution (current vs new)")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.xticks(x, [str(v) for v in DISCRETE_SCORES])
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_score_compare.png", dpi=150)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="temp/evaluation_results_*.csv")
    ap.add_argument("--n-files", type=int, default=5)
    ap.add_argument("--out-dir", default="temp/score_dist_tune_out")

    ap.add_argument("--params-json", default="",
                    help="新パラメータをJSONで指定する場合のパス（未指定なら自動提案）")

    # A確定仕様
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--min-discrete-ratio", type=float, default=0.95)

    # face系はデフォルト自動提案をしない（極端寄り抑制は“監視”）
    ap.add_argument("--apply-face-auto", action="store_true",
                    help="face系にも自動閾値提案を適用する（デフォルトはOFF推奨）")
    ap.add_argument("--face-saturation-warn", type=float, default=0.80,
                    help="face系で(0+1)飽和率がこの値を超えたら警告")

    args = ap.parse_args()

    latest_files = find_latest_files(args.input_glob, args.n_files)
    info("using files (latest first):")
    for p in latest_files:
        info(f"  - {p}")

    df = read_concat_csv(latest_files)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # JSON params 読み込み（あれば優先）
    # 形式:
    # {
    #   "contrast": {"thresholds": [..4..]},
    #   "sharpness": {"thresholds": [..4..]}
    # }
    # metricキーは "contrast" のように score/raw の接頭辞部分
    json_params: Dict[str, Thresholds] = {}
    if args.params_json:
        pj = Path(args.params_json)
        with pj.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for metric, spec in data.items():
            if "thresholds" not in spec:
                raise ValueError(f"params_json metric '{metric}' missing 'thresholds'")
            json_params[metric] = Thresholds.from_list(spec["thresholds"])

    comparison_rows: List[Dict] = []
    metric_summary: List[Dict] = []
    chosen_params_out: Dict[str, Dict] = {}

    for score_col in TARGET_SCORE_COLS:
        metric = score_col.replace("_score", "")
        raw_col, raw_note = resolve_raw_col(df, score_col)
        if raw_col is None:
            warn(f"Skip '{metric}': missing required raw column (raw resolve failed)")
            continue

        # 必須カラム存在チェック
        ok, reason = validate_score_column(
            df, score_col, raw_col,
            min_samples=args.min_samples,
            min_discrete_ratio=args.min_discrete_ratio,
        )
        if not ok:
            warn(f"Skip '{metric}': {reason}")
            continue

        is_tech = score_col in TECH_SCORE_COLS
        is_face = score_col in FACE_SCORE_COLS

        raw = df[raw_col]
        cur_score = df[score_col]

        # 現状分布
        cur_ratios = compute_score_ratios(cur_score)
        cur_counts = compute_score_counts(cur_score)
        cur_sat = saturation_ratio(cur_ratios)
        cur_l1 = tech_target_l1(cur_ratios) if is_tech else None
        cur_flags = tech_target_flags(cur_ratios) if is_tech else None

        # 新閾値決定（json > auto）
        thr_source = "none"
        thr: Optional[Thresholds] = None

        if metric in json_params:
            thr = json_params[metric]
            thr_source = "json"
        else:
            if is_tech:
                thr = propose_thresholds_for_target_distribution(raw)
                thr_source = "auto_target_quantiles_12.5_32.5_62.5_87.5"
            elif is_face and args.apply_face_auto:
                thr = propose_thresholds_for_target_distribution(raw)
                thr_source = "auto_target_quantiles_12.5_32.5_62.5_87.5"
            else:
                thr = None
                thr_source = "skip_auto"

        # 新スコア計算（thrが無い場合は current と同一にして出力は継続）
        raw_num = coerce_numeric(raw)
        # ★ここで raw の向きを統一する（rawは「高いほど良い」）
        if metric in ("noise", "face_noise") and raw_note == "fallback:sigma_used":
            raw_num = -raw_num

        if thr is not None:
            new_score = raw_num.map(lambda x: score_from_raw(x, thr))
            chosen_params_out[metric] = {
                "raw_col": raw_col,
                "score_col": score_col,
                "thresholds": thr.to_list(),
                "source": thr_source,
            }
        else:
            new_score = coerce_numeric(cur_score)
            chosen_params_out[metric] = {
                "raw_col": raw_col,
                "score_col": score_col,
                "thresholds": None,
                "source": thr_source,
            }

        new_ratios = compute_score_ratios(new_score)
        new_counts = compute_score_counts(new_score)
        new_sat = saturation_ratio(new_ratios)
        new_l1 = tech_target_l1(new_ratios) if is_tech else None
        new_flags = tech_target_flags(new_ratios) if is_tech else None

        # face系の極端寄り監視
        if is_face and (cur_sat >= args.face_saturation_warn or new_sat >= args.face_saturation_warn):
            warn(f"Face metric '{metric}' extreme? saturation(0+1): current={cur_sat:.3f}, new={new_sat:.3f} (warn>= {args.face_saturation_warn})")

        # long 形式比較（分布）
        for sv in DISCRETE_SCORES:
            comparison_rows.append({
                "metric": metric,
                "which": "current",
                "score": sv,
                "count": cur_counts.get(sv, 0),
                "ratio": cur_ratios.get(sv, 0.0),
            })
            comparison_rows.append({
                "metric": metric,
                "which": "new",
                "score": sv,
                "count": new_counts.get(sv, 0),
                "ratio": new_ratios.get(sv, 0.0),
            })

        # metric summary
        row = {
            "metric": metric,
            "type": "tech" if is_tech else ("face" if is_face else "other"),
            "score_col": score_col,
            "raw_col": raw_col,
            "threshold_source": thr_source,
            "thresholds": thr.to_list() if thr is not None else "",
            "current_saturation_0plus1": cur_sat,
            "new_saturation_0plus1": new_sat,
            "current_discrete_ratio": float(discrete_ratio(cur_score)),
            "new_discrete_ratio": float(discrete_ratio(new_score)),
            "current_in_range_ratio": float(in_range_ratio(cur_score)),
            "new_in_range_ratio": float(in_range_ratio(new_score)),
        }

        if is_tech:
            row["current_tech_target_l1"] = float(cur_l1)
            row["new_tech_target_l1"] = float(new_l1)
            # 各スコア帯の目標レンジ内/外
            for sv in DISCRETE_SCORES:
                row[f"current_target_flag_{sv}"] = cur_flags[sv]
                row[f"new_target_flag_{sv}"] = new_flags[sv]
        else:
            row["current_tech_target_l1"] = ""
            row["new_tech_target_l1"] = ""
            for sv in DISCRETE_SCORES:
                row[f"current_target_flag_{sv}"] = ""
                row[f"new_target_flag_{sv}"] = ""

        row["raw_resolve"] = raw_note
        metric_summary.append(row)

        # plots
        make_plots(plots_dir, metric, raw, cur_score, new_score)

        # console quick view
        info("-" * 72)
        info(f"[METRIC] {metric}")
        info(f"  source={thr_source} thresholds={thr.to_list() if thr is not None else None}")
        info("  score : current_ratio -> new_ratio (delta)")
        for sv in DISCRETE_SCORES:
            c = cur_ratios.get(sv, 0.0)
            n = new_ratios.get(sv, 0.0)
            d = n - c
            info(f"  {sv:>4} : {c:6.3f} -> {n:6.3f} ({d:+6.3f})")
        if is_tech:
            info(f"  tech_target_l1: {cur_l1:.6f} -> {new_l1:.6f}")

    if not comparison_rows:
        warn("No valid metrics produced outputs (all skipped by checks).")
        return 2

    # save params
    params_path = out_dir / "new_params_used.json"
    with params_path.open("w", encoding="utf-8") as f:
        json.dump(chosen_params_out, f, ensure_ascii=False, indent=2)

    # save distribution long/wide
    comp_df = pd.DataFrame(comparison_rows)
    long_csv = out_dir / "score_distribution_long.csv"
    comp_df.to_csv(long_csv, index=False)

    cur_df = comp_df[comp_df["which"] == "current"].rename(columns={"count": "current_count", "ratio": "current_ratio"}).drop(columns=["which"])
    new_df = comp_df[comp_df["which"] == "new"].rename(columns={"count": "new_count", "ratio": "new_ratio"}).drop(columns=["which"])
    wide = pd.merge(cur_df, new_df, on=["metric", "score"], how="outer").fillna(0)
    wide["delta_ratio"] = wide["new_ratio"] - wide["current_ratio"]
    wide_csv = out_dir / "score_distribution_wide.csv"
    wide.to_csv(wide_csv, index=False)

    # save metric summary
    summary_df = pd.DataFrame(metric_summary)
    summary_csv = out_dir / "metric_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    info("=" * 80)
    info("[DONE]")
    info(f"  out_dir : {out_dir}")
    info(f"  params  : {params_path}")
    info(f"  long    : {long_csv}")
    info(f"  wide    : {wide_csv}")
    info(f"  summary : {summary_csv}")
    info(f"  plots   : {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
