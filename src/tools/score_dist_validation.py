# src/tools/score_dist_validation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


DISCRETE_SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]
DISCRETE_SET = set(DISCRETE_SCORES)


@dataclass(frozen=True)
class Criteria7015:
    """
    #701-5: 閾値調整後の分布健全性チェックの SSOT
    """

    sat_max: float = 0.40
    accepted_delta_max: float = 0.10
    discrete_min: float = 0.95
    in_range_min: float = 0.98
    require_target_l1_improve: bool = False
    direction_conflict_fail: bool = False


def _is_nan(x: Any) -> bool:
    try:
        return bool(float(x) != float(x))
    except Exception:
        return True


def validate_row_701_5(
    row: Dict[str, Any],
    *,
    criteria: Criteria7015,
) -> Tuple[bool, List[str]]:
    """
    metric_summary の1行(row)を検証する。
    戻り値: (pass, reasons)
    """
    reasons: List[str] = []

    # 1) saturation
    try:
        new_sat = float(row.get("new_saturation_0plus1"))
        if new_sat >= criteria.sat_max:
            reasons.append(f"sat_0plus1>= {criteria.sat_max} (new={new_sat:.3f})")
    except Exception:
        reasons.append("sat_0plus1 missing/invalid")

    # 2) accepted delta
    cur_acc = row.get("current_accepted_ratio")
    new_acc = row.get("new_accepted_ratio")
    if _is_nan(cur_acc) or _is_nan(new_acc) or (cur_acc == "") or (new_acc == ""):
        reasons.append("accepted_ratio missing (ranking/acceptance?)")
    else:
        try:
            d = abs(float(new_acc) - float(cur_acc))
            if d > criteria.accepted_delta_max:
                reasons.append(
                    f"accepted_delta> {criteria.accepted_delta_max} (abs_delta={d:.3f})"
                )
        except Exception:
            reasons.append("accepted_ratio invalid")

    # 3) direction consistency
    dfin = str(row.get("direction_final", ""))
    if dfin == "conflict" and criteria.direction_conflict_fail:
        reasons.append("direction_conflict")

    # 4) discrete ratio (new)
    try:
        new_disc = float(row.get("new_discrete_ratio"))
        if new_disc < criteria.discrete_min:
            reasons.append(
                f"discrete_ratio< {criteria.discrete_min} (new={new_disc:.3f})"
            )
    except Exception:
        reasons.append("new_discrete_ratio missing/invalid")

    # 5) in-range ratio (new)
    try:
        new_ir = float(row.get("new_in_range_ratio"))
        if new_ir < criteria.in_range_min:
            reasons.append(
                f"in_range_ratio< {criteria.in_range_min} (new={new_ir:.3f})"
            )
    except Exception:
        reasons.append("new_in_range_ratio missing/invalid")

    # 6) tech_target_l1 improve (tech only)
    if criteria.require_target_l1_improve and row.get("type") == "tech":
        try:
            cur_l1 = float(row.get("current_tech_target_l1"))
            new_l1 = float(row.get("new_tech_target_l1"))
            if new_l1 > cur_l1:
                reasons.append(
                    f"tech_target_l1_regressed (cur={cur_l1:.6f}, new={new_l1:.6f})"
                )
        except Exception:
            reasons.append("tech_target_l1 missing/invalid")

    passed = len(reasons) == 0
    return passed, reasons
