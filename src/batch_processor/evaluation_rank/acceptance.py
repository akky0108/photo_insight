# src/batch_processor/evaluation_rank/acceptance.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

Row = Dict[str, Any]


# ==============================
# ユーティリティ
# ==============================

def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def format_score(x: float) -> float:
    return round(float(x), 2)


def percentile(values: List[Any], p: float) -> float:
    """
    numpy なし percentile（線形補間）。
    - values に "" / None / 非数値が混ざっても壊れない
    - p: 0〜100
    """
    xs: List[float] = []
    for v in values:
        if v in ("", None):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        xs.append(fv)

    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]

    xs.sort()
    p = max(0.0, min(100.0, float(p)))
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def extract_contrib(prefix: str, r: Row) -> Dict[str, Any]:
    """
    prefix は "contrib_tech_" / "contrib_face_" / "contrib_comp_" を想定。
    prefix を外したキー -> 値 の Dict にして返す。
    """
    out: Dict[str, Any] = {}
    for k, v in r.items():
        if not isinstance(k, str):
            continue
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _bool(value: Any) -> bool:
    """
    CSV の "TRUE"/"False"/"1"/"" 対応のための bool 正規化。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "t", "yes", "y"):
            return True
        if v in ("0", "false", "f", "no", "n", ""):
            return False
    try:
        return bool(int(value))
    except Exception:
        return False


def safe_int_flag(value: Any) -> int:
    """
    CSV由来の 0/1, True/False, "TRUE"/"False" を 0/1 に正規化する。
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


def reliability_score(row: Row) -> float:
    """
    コントラクトに基づく「測定信頼度」(0..5)。
    score は落とさず主指標として使い、同点付近の tie-break に使う。
    """
    s = 0.0

    if str(row.get("noise_eval_status", "")).strip().lower() == "ok":
        s += 1.0
    if str(row.get("exposure_eval_status", "")).strip().lower() == "ok":
        s += 1.0
    if str(row.get("sharpness_eval_status", "")).strip().lower() == "ok":
        s += 1.0
    if str(row.get("contrast_eval_status", "")).strip().lower() == "ok":
        s += 1.0

    # blurriness: success True/False（キー揺れ吸収）
    blur_ok = (
        _bool(row.get("success"))
        or _bool(row.get("blurriness_success"))
        or _bool(row.get("blurriness_ok"))
        or _bool(row.get("blurriness_eval_ok"))
    )
    if blur_ok:
        s += 1.0

    return s


def overall_sort_key(row: Row) -> Tuple[float, float, float, float, float, str]:
    """
    overall_score を軸にしつつ、サブスコア＋信頼度で tie-break するソートキー。
    """
    overall = safe_float(row.get("overall_score"))
    face = safe_float(row.get("score_face"))
    comp = safe_float(row.get("score_composition"))
    tech = safe_float(row.get("score_technical"))
    rel = reliability_score(row)
    fname = str(row.get("file_name") or row.get("filename") or "")
    return (overall, face, comp, tech, rel, fname)


# ==============================
# accepted_reason / secondary_accept_reason の構築
# ==============================

def build_accepted_reason(
    *,
    category: str,
    rank: int,
    max_accept: int,
    threshold: float,
    overall: float,
    accepted_flag: int,
    score_technical: Optional[float],
    score_face: Optional[float],
    score_composition: Optional[float],
    contrib_tech: Dict[str, Any],
    contrib_face: Dict[str, Any],
    contrib_comp: Dict[str, Any],
    top_n: int = 3,
    accept_group: Optional[str] = None,
) -> str:
    """
    accepted_reason を 1列にまとめる（一次採用用）。
    - accepted_flag=1: contrib 上位Nだけで短文化
    - accepted_flag=0: 空文字（運用互換）
    """
    def _to_float(d: Dict[str, Any]) -> Dict[str, float]:
        return {k: safe_float(v) for k, v in d.items()}

    def _top_items(d: Dict[str, float], n: int) -> List[Tuple[str, float]]:
        items = [(k, v) for k, v in d.items() if v is not None]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    if accepted_flag != 1:
        return ""

    t = _to_float(contrib_tech)
    f = _to_float(contrib_face)
    c = _to_float(contrib_comp)

    merged: Dict[str, float] = {}
    merged.update({f"tech.{k}": v for k, v in t.items()})
    merged.update({f"face.{k}": v for k, v in f.items()})
    merged.update({f"comp.{k}": v for k, v in c.items()})

    top = _top_items(merged, top_n)
    top_txt = ", ".join(f"{k}={format_score(v)}" for k, v in top) if top else ""

    group_txt = f" group={accept_group}" if accept_group else ""
    return (
        f"{category}{group_txt} "
        f"rank={rank}/{max_accept} "
        f"thr={format_score(threshold)} "
        f"overall={format_score(overall)} top={top_txt}"
    )


def build_secondary_reason(
    *,
    category: str,
    accept_group: Optional[str],
    sec_thr: float,
    overall: float,
    score_face: float,
    score_composition: float,
    score_technical: float,
    prefix: str = "SEC:",
) -> str:
    """
    secondary_accept_reason 用（Lightroomキーワードにもそのまま使える形）
    """
    g = f" group={accept_group}" if accept_group else ""
    return (
        f"{prefix}{category}{g} "
        f"sec_thr={format_score(sec_thr)} "
        f"overall={format_score(overall)} "
        f"f={format_score(score_face)} "
        f"c={format_score(score_composition)} "
        f"t={format_score(score_technical)}"
    )


# ==============================
# 救済ルール（portrait 用）
# ==============================

def is_composition_rescue_candidate(row: Row) -> bool:
    """
    構図が強いポートレート用の救済条件。
    """
    if row.get("category") != "portrait":
        return False
    if not _bool(row.get("face_detected")):
        return False

    comp = safe_float(row.get("score_composition"))
    tech = safe_float(row.get("score_technical"))

    eye = safe_float(row.get("eye_contact_score"))
    direction = safe_float(row.get("face_direction_score"))

    lead = safe_float(row.get("lead_room_score"))
    if lead == 0.0:
        lead = safe_float(row.get("contrib_comp_lead_room_score"))

    if comp < 70.0:
        return False
    if tech < 45.0:
        return False

    if eye >= 0.80:
        return True

    if eye < 0.60 and direction >= 0.80 and lead >= 0.80:
        return True

    return False


def is_face_rescue_candidate(row: Row, accept_thr: float) -> bool:
    """
    顔が一番頑張っているボーダーショットの救済条件。
    """
    if row.get("category") != "portrait":
        return False

    overall = safe_float(row.get("overall_score"))
    if overall < 40.0 or overall >= accept_thr:
        return False

    comp = safe_float(row.get("score_composition"))
    face = safe_float(row.get("score_face"))
    tech = safe_float(row.get("score_technical"))

    if face < 40.0:
        return False

    if face < comp + 5.0 and face < tech + 5.0:
        return False

    return True


def apply_rescue_logic(
    row: Row,
    *,
    accepted_flag: int,
    secondary_accept_flag: int,
    accept_thr: float,
    sec_thr: float,
    accepted_reason: str,
    secondary_accept_reason: str,
) -> Tuple[int, int, str, str]:
    """
    一次/二次判定の後に、portrait 用の救済を適用。
    - 救済は「二次採用」なので secondary_accept_reason に書く
    """
    if accepted_flag:
        return accepted_flag, secondary_accept_flag, accepted_reason, secondary_accept_reason

    if row.get("category") != "portrait":
        return accepted_flag, secondary_accept_flag, accepted_reason, secondary_accept_reason

    overall = safe_float(row.get("overall_score"))
    face = safe_float(row.get("score_face"))
    comp = safe_float(row.get("score_composition"))
    tech = safe_float(row.get("score_technical"))
    group = row.get("accept_group")

    if is_composition_rescue_candidate(row):
        secondary_accept_flag = 1
        secondary_accept_reason = (
            f"SEC-RESCUE:portrait(comp) group={group} "
            f"thr={format_score(accept_thr)} "
            f"overall={format_score(overall)} "
            f"f={format_score(face)} c={format_score(comp)} t={format_score(tech)}"
        )
        return accepted_flag, secondary_accept_flag, "", secondary_accept_reason

    if is_face_rescue_candidate(row, accept_thr):
        secondary_accept_flag = 1
        secondary_accept_reason = (
            f"SEC-RESCUE:portrait(face) group={group} "
            f"thr={format_score(accept_thr)} "
            f"overall={format_score(overall)} "
            f"f={format_score(face)} c={format_score(comp)} t={format_score(tech)}"
        )
        return accepted_flag, secondary_accept_flag, "", secondary_accept_reason

    return accepted_flag, secondary_accept_flag, accepted_reason, secondary_accept_reason


# ==============================
# ルール定義
# ==============================

@dataclass(frozen=True)
class AcceptRules:
    portrait_percentile: float = 70.0
    portrait_secondary_percentile: float = 60.0

    non_face_percentile: float = 80.0
    non_face_secondary_percentile: float = 70.0

    # ★ Green を「全体の何%」にするか
    green_ratio_total: float = 0.15  # 15%

    # ★ 最低でも何枚はGreenにするか（少数データ対策）
    green_min_total: int = 1

    top_flag_ratio: float = 0.40  # flag(=Blue候補) は従来通り


# ==============================
# エンジン本体
# ==============================

class AcceptanceEngine:
    def __init__(self, rules: Optional[AcceptRules] = None) -> None:
        self.rules = rules or AcceptRules()

    def _assign_accept_group(self, r: Row) -> str:
        base_category = "portrait" if _bool(r.get("face_detected")) else "non_face"
        if base_category == "non_face":
            return "non_face"

        group_id = str(r.get("group_id") or "").strip().upper()
        subgroup_id = str(r.get("subgroup_id") or "").strip()
        shot_type = str(r.get("shot_type") or "").strip()

        if group_id in ("A", "B"):
            if subgroup_id in ("1", "2"):
                return f"{group_id}-{subgroup_id}"
            return group_id or "portrait_misc"

        if shot_type in ("face_only", "upper_body"):
            return "A-1" if not subgroup_id else f"A-{subgroup_id}"

        if shot_type in ("full_body", "seated"):
            return "B-1" if not subgroup_id else f"B-{subgroup_id}"

        return "portrait_misc"

    def assign_category(self, rows: List[Row]) -> None:
        for r in rows:
            face = _bool(r.get("face_detected"))
            r["category"] = "portrait" if face else "non_face"
            r["accept_group"] = self._assign_accept_group(r)

    def compute_thresholds(self, rows: List[Row]) -> Dict[str, float]:
        portrait_rows = [r for r in rows if r.get("category") == "portrait"]
        non_face_rows = [r for r in rows if r.get("category") == "non_face"]

        portrait_scores = [r.get("overall_score") for r in portrait_rows]
        non_face_scores = [r.get("overall_score") for r in non_face_rows]

        if portrait_scores:
            portrait_thr = percentile(portrait_scores, self.rules.portrait_percentile)
            portrait_sec_thr = percentile(portrait_scores, self.rules.portrait_secondary_percentile)
        else:
            portrait_thr = portrait_sec_thr = 0.0

        # sec <= primary を保証（安全ガード）
        if portrait_sec_thr > portrait_thr:
            portrait_sec_thr = portrait_thr

        if non_face_scores:
            non_face_thr = percentile(non_face_scores, self.rules.non_face_percentile)
            non_face_sec_thr = percentile(non_face_scores, self.rules.non_face_secondary_percentile)
        else:
            non_face_thr = non_face_sec_thr = 0.0

        if non_face_sec_thr > non_face_thr:
            non_face_sec_thr = non_face_thr

        return {
            "portrait": portrait_thr,
            "portrait_secondary": portrait_sec_thr,
            "non_face": non_face_thr,
            "non_face_secondary": non_face_sec_thr,
        }

    def apply_accepted_flags(self, rows: List[Row]) -> Dict[str, float]:
        self.assign_category(rows)
        thresholds = self.compute_thresholds(rows)

        # 初期化
        for r in rows:
            r["accepted_flag"] = 0
            r["secondary_accept_flag"] = 0
            r["accepted_reason"] = ""
            r["secondary_accept_reason"] = ""

        # 全体Green枠
        total_n = len(rows)
        green_total = max(
            int(math.ceil(total_n * float(self.rules.green_ratio_total))),
            int(self.rules.green_min_total),
        )

        # 全体ランキング
        sorted_all = sorted(rows, key=overall_sort_key, reverse=True)

        portrait_thr = thresholds["portrait"]
        portrait_sec_thr = thresholds["portrait_secondary"]
        non_face_thr = thresholds["non_face"]
        non_face_sec_thr = thresholds["non_face_secondary"]

        def _primary_ok(r: Row) -> bool:
            overall = safe_float(r.get("overall_score"))
            if r.get("category") == "portrait":
                return overall >= portrait_thr
            return overall >= non_face_thr

        def _secondary_ok(r: Row) -> bool:
            overall = safe_float(r.get("overall_score"))
            if r.get("category") == "portrait":
                return overall >= portrait_sec_thr
            return overall >= non_face_sec_thr

        # rank 用（表示用）
        portrait_rank = 0
        non_face_rank = 0

        accepted_count = 0

        # ---- PASS1: primary threshold を満たすものだけで埋める ----
        for r in sorted_all:
            if accepted_count >= green_total:
                break

            # category_rank を付ける（全体を見ながらカテゴリ内順位も持つ）
            if r.get("category") == "portrait":
                portrait_rank += 1
                r["category_rank"] = portrait_rank
            else:
                non_face_rank += 1
                r["category_rank"] = non_face_rank

            overall = safe_float(r.get("overall_score"))

            if _primary_ok(r):
                accepted_count += 1
                r["accepted_flag"] = 1
                r["accepted_reason"] = build_accepted_reason(
                    category=str(r.get("category") or ""),
                    rank=accepted_count,           # ★Green枠内順位
                    max_accept=green_total,        # ★全体15%枠
                    threshold=(portrait_thr if r.get("category") == "portrait" else non_face_thr),
                    overall=overall,
                    accepted_flag=1,
                    score_technical=safe_float(r.get("score_technical")),
                    score_face=(safe_float(r.get("score_face")) if r.get("category") == "portrait" else None),
                    score_composition=safe_float(r.get("score_composition")),
                    contrib_tech=extract_contrib("contrib_tech_", r),
                    contrib_face=extract_contrib("contrib_face_", r) if r.get("category") == "portrait" else {},
                    contrib_comp=extract_contrib("contrib_comp_", r),
                    top_n=3,
                    accept_group=r.get("accept_group"),
                )

        # ---- PASS2: まだ枠が余るなら secondary threshold で埋める ----
        if accepted_count < green_total:
            for r in sorted_all:
                if accepted_count >= green_total:
                    break
                if safe_int_flag(r.get("accepted_flag")) == 1:
                    continue

                overall = safe_float(r.get("overall_score"))
                if _secondary_ok(r):
                    accepted_count += 1
                    r["accepted_flag"] = 1
                    r["accepted_reason"] = (
                        f"ACC-SEC-FILL:{r.get('category')} "
                        f"rank={accepted_count}/{green_total} "
                        f"o={format_score(overall)} "
                        f"sec_thr={format_score(portrait_sec_thr if r.get('category')=='portrait' else non_face_sec_thr)}"
                    )

        # ---- PASS3: それでも余るなら「上から埋める」（極端ケースの保険） ----
        if accepted_count < green_total:
            for r in sorted_all:
                if accepted_count >= green_total:
                    break
                if safe_int_flag(r.get("accepted_flag")) == 1:
                    continue

                overall = safe_float(r.get("overall_score"))
                accepted_count += 1
                r["accepted_flag"] = 1
                r["accepted_reason"] = (
                    f"ACC-FILL:{r.get('category')} "
                    f"rank={accepted_count}/{green_total} "
                    f"o={format_score(overall)}"
                )

        # ---- secondary_accept_flag（Yellow用）は従来ロジックを維持 ----
        # 「Green枠に入らなかった」ものを secondary として付ける（閾値=percentile由来）
        for r in rows:
            if safe_int_flag(r.get("accepted_flag")) == 1:
                continue

            overall = safe_float(r.get("overall_score"))
            if r.get("category") == "portrait" and overall >= portrait_sec_thr:
                r["secondary_accept_flag"] = 1
                r["secondary_accept_reason"] = build_secondary_reason(
                    category="portrait",
                    accept_group=r.get("accept_group"),
                    sec_thr=portrait_sec_thr,
                    overall=overall,
                    score_face=safe_float(r.get("score_face")),
                    score_composition=safe_float(r.get("score_composition")),
                    score_technical=safe_float(r.get("score_technical")),
                    prefix="SEC:",
                )
                # A方式: accepted_reason を統一して見せる
                r["accepted_reason"] = str(r.get("secondary_accept_reason") or "")

            elif r.get("category") == "non_face" and overall >= non_face_sec_thr:
                r["secondary_accept_flag"] = 1
                r["secondary_accept_reason"] = (
                    f"SEC:non_face group={r.get('accept_group')} "
                    f"sec_thr={format_score(non_face_sec_thr)} "
                    f"overall={format_score(overall)} "
                    f"c={format_score(safe_float(r.get('score_composition')))} "
                    f"t={format_score(safe_float(r.get('score_technical')))}"
                )
                r["accepted_reason"] = str(r.get("secondary_accept_reason") or "")

        return thresholds


    def apply_top_flag(self, rows: List[Row]) -> None:
        if not rows:
            return
        sorted_rows = sorted(rows, key=overall_sort_key, reverse=True)
        top_n = max(1, int(len(sorted_rows) * float(self.rules.top_flag_ratio)))
        for i, r in enumerate(sorted_rows):
            r["flag"] = 1 if i < top_n else 0

    def run(self, rows: List[Row]) -> Dict[str, float]:
        thresholds = self.apply_accepted_flags(rows)
        self.apply_top_flag(rows)
        return thresholds
