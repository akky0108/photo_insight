#!/usr/bin/env python3
#tools/validate_metric_summary.py
# -*- coding: utf-8 -*-

"""
validate_metric_summary.py

metric_summary.csv を読み、退行/異常を自動判定する（CI向け）。

方針（A案）:
- FAIL（exit=2）は "tech" のみに適用（face は監視=WARN）
- ただし direction_final == "conflict" は契約違反として type に関係なく FAIL

FAIL 条件（退出コード 2）:
- (tech) new_saturation_0plus1 >= sat_limit_tech
- (tech) new_discrete_ratio < min_discrete_ratio
- (tech) new_in_range_ratio < min_in_range_ratio
- (all)  direction_final == "conflict"

WARN 条件（退出コードは 0 のまま）:
- (all)  delta_saturation_0plus1 > sat_delta_warn
- (tech) delta_tech_target_l1 > 0（悪化）
- (tech) new_target_match_ratio < target_match_warn
- (all)  current_accepted_ratio が欠損（rankingが読めていない等）
- (all)  direction_final == "unknown"（推定根拠が弱い）※ --warn-on-direction-unknown
- (face) new_saturation_0plus1 >= sat_limit_face_warn（顔の極端寄りを監視）

使い方:
  python tools/validate_metric_summary.py \
    --summary-csv temp/score_dist_tune_out/metric_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def _to_float(x: Any) -> Optional[float]:
    """Convert to float, treating NaN/None/'' as None."""
    try:
        if x is None:
            return None
        if isinstance(x, float) and (x != x):  # NaN
            return None
        if x == "":
            return None
        return float(x)
    except Exception:
        return None


def _to_str(x: Any) -> str:
    """Convert to str, treating NaN/None as ''."""
    if x is None:
        return ""
    if isinstance(x, float) and (x != x):  # NaN
        return ""
    return str(x)


def _is_true(x: Any) -> bool:
    """Loose bool parsing for columns like 0/1, 'true', 'False', etc."""
    if x is None:
        return False
    if isinstance(x, bool):
        return x
    s = _to_str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default="temp/score_dist_tune_out/metric_summary.csv")

    # hard fail thresholds (TECH only)
    ap.add_argument("--sat-limit-tech", type=float, default=0.4)
    ap.add_argument("--min-discrete-ratio", type=float, default=0.95)
    ap.add_argument("--min-in-range-ratio", type=float, default=0.98)

    # warnings
    ap.add_argument("--sat-delta-warn", type=float, default=0.10)
    ap.add_argument("--target-match-warn", type=float, default=0.60)  # tech only
    ap.add_argument("--warn-on-direction-unknown", action="store_true")

    # face monitoring (WARN only)
    ap.add_argument("--sat-limit-face-warn", type=float, default=0.90)

    # optional toggles
    ap.add_argument(
        "--fail-on-missing-accepted-ratio",
        action="store_true",
        help="If set, missing current_accepted_ratio becomes FAIL (default WARN).",
    )

    args = ap.parse_args()

    p = Path(args.summary_csv)
    if not p.exists():
        print(f"[FAIL] metric_summary not found: {p}", file=sys.stderr)
        return 2

    df = pd.read_csv(p)

    fails: list[str] = []
    warns: list[str] = []

    # columns may be missing depending on version; handle gracefully
    for _, r in df.iterrows():
        metric = _to_str(r.get("metric"))
        mtype = _to_str(r.get("type")).strip().lower()

        # core metrics
        new_sat = _to_float(r.get("new_saturation_0plus1"))
        new_disc = _to_float(r.get("new_discrete_ratio"))
        new_rng = _to_float(r.get("new_in_range_ratio"))

        d_final = _to_str(r.get("direction_final")).strip().lower()
        d_sat = _to_float(r.get("delta_saturation_0plus1"))

        # -----------------
        # FAIL rules (A: TECH only, except conflict)
        # -----------------
        if d_final == "conflict":
            fails.append(f"{metric}: direction conflict (direction_final=conflict)")

        if mtype == "tech":
            if new_sat is not None and new_sat >= args.sat_limit_tech:
                fails.append(
                    f"{metric}: saturation too high (new={new_sat:.3f} >= {args.sat_limit_tech})"
                )

            if new_disc is not None and new_disc < args.min_discrete_ratio:
                fails.append(
                    f"{metric}: discrete too low (new={new_disc:.3f} < {args.min_discrete_ratio})"
                )

            if new_rng is not None and new_rng < args.min_in_range_ratio:
                fails.append(
                    f"{metric}: in_range too low (new={new_rng:.3f} < {args.min_in_range_ratio})"
                )

        # -----------------
        # WARN rules
        # -----------------
        if d_sat is not None and d_sat > args.sat_delta_warn:
            warns.append(
                f"{metric}: saturation increased (delta={d_sat:+.3f} > {args.sat_delta_warn:+.3f})"
            )

        cur_acc = _to_float(r.get("current_accepted_ratio"))
        if cur_acc is None:
            msg = f"{metric}: current_accepted_ratio missing (ranking not available?)"
            if args.fail_on_missing_accepted_ratio:
                fails.append(msg)
            else:
                warns.append(msg)

        if d_final == "unknown" and args.warn_on_direction_unknown:
            warns.append(f"{metric}: direction_final unknown (insufficient evidence)")

        # face monitoring only (WARN)
        if mtype == "face":
            if new_sat is not None and new_sat >= args.sat_limit_face_warn:
                warns.append(
                    f"{metric}: face saturation high (new={new_sat:.3f} >= {args.sat_limit_face_warn})"
                )

        # tech-only warnings
        if mtype == "tech":
            d_l1 = _to_float(r.get("delta_tech_target_l1"))
            if d_l1 is not None and d_l1 > 0:
                warns.append(f"{metric}: tech_target_l1 worsened (delta={d_l1:+.6f} > 0)")

            tmr = _to_float(r.get("new_target_match_ratio"))
            if tmr is not None and tmr < args.target_match_warn:
                warns.append(
                    f"{metric}: target_match_ratio low (new={tmr:.2f} < {args.target_match_warn})"
                )

    if warns:
        print("[WARN] ---", file=sys.stderr)
        for w in warns:
            print(f"[WARN] {w}", file=sys.stderr)

    if fails:
        print("[FAIL] ---", file=sys.stderr)
        for f in fails:
            print(f"[FAIL] {f}", file=sys.stderr)
        return 2

    print("[OK] metric_summary validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
