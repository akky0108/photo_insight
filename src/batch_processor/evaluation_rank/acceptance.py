# src/batch_processor/evaluation_rank/acceptance.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# accepted_flag（分布ベース）
DEFAULT_ACCEPT_RULES = {
    "portrait": {"percentile": 70, "max_accept": 5},
    "non_face": {"percentile": 80, "max_accept": 3},
}

# flag（従来通り：全体上位比率）
DEFAULT_TOP_FLAG_RATIO = 0.35


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def format_score(x: float) -> float:
    return round(float(x), 2)


def percentile(values: List[float], p: float) -> float:
    """
    numpy無しpercentile（線形補間）。
    p: 0-100
    """
    xs = sorted([float(x) for x in values if x is not None])
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]

    p = max(0.0, min(100.0, float(p)))
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def extract_contrib(prefix: str, r: Dict[str, Any]) -> Dict[str, Any]:
    """
    prefix は "contrib_tech_" / "contrib_face_" / "contrib_comp_" を想定
    """
    out: Dict[str, Any] = {}
    for k, v in r.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


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
) -> str:
    """
    accepted_reason を 1列にまとめる。
    - accepted_flag=1: contrib 上位Nだけで短文化
    - accepted_flag=0: フル（デバッグ用に残す）
    """
    def _to_float(d: Dict[str, Any]) -> Dict[str, float]:
        return {k: safe_float(v) for k, v in d.items()}

    def _top_items(d: Dict[str, float], n: int) -> List[Tuple[str, float]]:
        items = [(k, v) for k, v in d.items() if v is not None]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    # accepted_flag=1 は短文化（必要情報だけ）
    if accepted_flag == 1:
        t = _to_float(contrib_tech)
        f = _to_float(contrib_face)
        c = _to_float(contrib_comp)

        merged: Dict[str, float] = {}
        merged.update({f"tech.{k}": v for k, v in t.items()})
        merged.update({f"face.{k}": v for k, v in f.items()})
        merged.update({f"comp.{k}": v for k, v in c.items()})

        top = _top_items(merged, top_n)
        top_txt = ", ".join([f"{k}={format_score(v)}" for k, v in top]) if top else ""

        return (
            f"{category} rank={rank}/{max_accept} thr={format_score(threshold)} "
            f"overall={format_score(overall)} top={top_txt}"
        )

    # accepted_flag=0 は空にする運用（現状互換）
    return ""


@dataclass(frozen=True)
class AcceptRules:
    portrait_percentile: float = DEFAULT_ACCEPT_RULES["portrait"]["percentile"]
    portrait_max_accept: int = DEFAULT_ACCEPT_RULES["portrait"]["max_accept"]
    non_face_percentile: float = DEFAULT_ACCEPT_RULES["non_face"]["percentile"]
    non_face_max_accept: int = DEFAULT_ACCEPT_RULES["non_face"]["max_accept"]
    top_flag_ratio: float = DEFAULT_TOP_FLAG_RATIO


class AcceptanceEngine:
    """
    cleanup() でやっている
    - category 付与
    - accepted_flag 計算
    - accepted_reason 生成
    - flag 付与
    をまとめて担当する
    """

    def __init__(self, rules: Optional[AcceptRules] = None):
        self.rules = rules or AcceptRules()

    def assign_category(self, rows: List[Dict[str, Any]]) -> None:
        for r in rows:
            face = bool(r.get("face_detected"))
            r["category"] = "portrait" if face else "non_face"

    def compute_thresholds(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        portrait_rows = [r for r in rows if r.get("category") == "portrait"]
        non_face_rows = [r for r in rows if r.get("category") == "non_face"]

        portrait_scores = [safe_float(r.get("overall_score")) for r in portrait_rows]
        non_face_scores = [safe_float(r.get("overall_score")) for r in non_face_rows]

        portrait_thr = percentile(portrait_scores, self.rules.portrait_percentile) if portrait_scores else 0.0
        non_face_thr = percentile(non_face_scores, self.rules.non_face_percentile) if non_face_scores else 0.0

        return {"portrait": portrait_thr, "non_face": non_face_thr}

    def apply_accepted_flags(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        rows を in-place で更新する。
        return: thresholds
        """
        self.assign_category(rows)
        th = self.compute_thresholds(rows)

        portrait_rows = [r for r in rows if r["category"] == "portrait"]
        non_face_rows = [r for r in rows if r["category"] == "non_face"]

        portrait_rows.sort(key=lambda r: safe_float(r.get("overall_score")), reverse=True)
        non_face_rows.sort(key=lambda r: safe_float(r.get("overall_score")), reverse=True)

        # portrait
        for i, r in enumerate(portrait_rows):
            rank = i + 1
            overall = safe_float(r.get("overall_score"))
            ok = (i < self.rules.portrait_max_accept) and (overall >= th["portrait"])
            r["accepted_flag"] = int(ok)

            r["accepted_reason"] = build_accepted_reason(
                category="portrait",
                rank=rank,
                max_accept=self.rules.portrait_max_accept,
                threshold=th["portrait"],
                overall=overall,
                accepted_flag=r["accepted_flag"],
                score_technical=safe_float(r.get("score_technical")),
                score_face=safe_float(r.get("score_face")),
                score_composition=safe_float(r.get("score_composition")),
                contrib_tech=extract_contrib("contrib_tech_", r),
                contrib_face=extract_contrib("contrib_face_", r),
                contrib_comp=extract_contrib("contrib_comp_", r),
                top_n=3,
            )

        # non_face
        for i, r in enumerate(non_face_rows):
            rank = i + 1
            overall = safe_float(r.get("overall_score"))
            ok = (i < self.rules.non_face_max_accept) and (overall >= th["non_face"])
            r["accepted_flag"] = int(ok)

            r["accepted_reason"] = build_accepted_reason(
                category="non_face",
                rank=rank,
                max_accept=self.rules.non_face_max_accept,
                threshold=th["non_face"],
                overall=overall,
                accepted_flag=r["accepted_flag"],
                score_technical=safe_float(r.get("score_technical")),
                score_face=None,
                score_composition=safe_float(r.get("score_composition")),
                contrib_tech=extract_contrib("contrib_tech_", r),
                contrib_face={},
                contrib_comp=extract_contrib("contrib_comp_", r),
                top_n=3,
            )

        return th

    def apply_top_flag(self, rows: List[Dict[str, Any]]) -> None:
        rows.sort(key=lambda r: safe_float(r.get("overall_score")), reverse=True)
        top_n = max(1, int(len(rows) * float(self.rules.top_flag_ratio)))
        for i, r in enumerate(rows):
            r["flag"] = 1 if i < top_n else 0

    def run(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        rows を in-place で更新して返す
        """
        thresholds = self.apply_accepted_flags(rows)
        self.apply_top_flag(rows)
        return thresholds
