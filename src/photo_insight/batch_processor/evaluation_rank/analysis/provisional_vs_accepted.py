# src/batch_processor/evaluation_rank/analysis/provisional_vs_accepted.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from collections import defaultdict


def _i01(v: Any) -> int:
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        return 1 if str(v).strip().lower() in ("1", "true", "t", "yes", "y") else 0


def _safe_float(v: Any) -> float:
    try:
        if v in ("", None):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d not in (0.0, 0) else 0.0


@dataclass(frozen=True)
class ProvisionalVsAcceptedSummaryRow:
    category: str
    accept_group: str
    percent: float

    total: int
    accepted: int
    provisional: int
    overlap: int
    accepted_not_top: int
    top_not_accepted: int

    # ---- rates (母数 = total) ----
    accepted_rate: float
    provisional_rate: float
    overlap_rate: float
    accepted_not_top_rate: float
    top_not_accepted_rate: float

    # ---- alignment metrics ----
    precision: float   # overlap / provisional
    recall: float      # overlap / accepted
    f1: float

    # ---- score stats ----
    mean_overall: float
    mean_face: float
    mean_comp: float
    mean_tech: float


def build_provisional_vs_accepted_summary(
    rows: Iterable[Dict[str, Any]],
    *,
    percent_key: str = "provisional_top_percent",
    prov_flag_key: str = "provisional_top_percent_flag",
    accepted_key: str = "accepted_flag",
) -> Tuple[List[ProvisionalVsAcceptedSummaryRow], Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    cat_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    all_rows: List[Dict[str, Any]] = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        all_rows.append(r)
        cat = str(r.get("category") or "").strip().lower() or "unknown"
        grp = str(r.get("accept_group") or "").strip().lower() or "unknown"
        buckets[(cat, grp)].append(r)
        cat_buckets[cat].append(r)

    def _summarize(cat: str, grp: str, items: List[Dict[str, Any]]) -> ProvisionalVsAcceptedSummaryRow:
        total = len(items)

        # percent は代表値として先頭から拾う（全行同一想定だが、違っても落ちない）
        try:
            p = float(items[0].get(percent_key) or 0.0) if total > 0 else 0.0
        except Exception:
            p = 0.0

        if total == 0:
            return ProvisionalVsAcceptedSummaryRow(
                category=cat, accept_group=grp, percent=p,
                total=0, accepted=0, provisional=0, overlap=0,
                accepted_not_top=0, top_not_accepted=0,
                accepted_rate=0.0, provisional_rate=0.0, overlap_rate=0.0,
                accepted_not_top_rate=0.0, top_not_accepted_rate=0.0,
                precision=0.0, recall=0.0, f1=0.0,
                mean_overall=0.0, mean_face=0.0, mean_comp=0.0, mean_tech=0.0,
            )

        accepted = provisional = overlap = 0
        sum_overall = sum_face = sum_comp = sum_tech = 0.0

        for rr in items:
            af = _i01(rr.get(accepted_key, 0))
            pf = _i01(rr.get(prov_flag_key, 0))
            accepted += af
            provisional += pf
            if af and pf:
                overlap += 1

            sum_overall += _safe_float(rr.get("overall_score"))
            sum_face += _safe_float(rr.get("score_face"))
            sum_comp += _safe_float(rr.get("score_composition"))
            sum_tech += _safe_float(rr.get("score_technical"))

        accepted_not_top = accepted - overlap
        top_not_accepted = provisional - overlap

        denom_total = float(total)
        accepted_rate = _safe_div(accepted, denom_total)
        provisional_rate = _safe_div(provisional, denom_total)
        overlap_rate = _safe_div(overlap, denom_total)
        accepted_not_top_rate = _safe_div(accepted_not_top, denom_total)
        top_not_accepted_rate = _safe_div(top_not_accepted, denom_total)

        precision = _safe_div(overlap, float(provisional))
        recall = _safe_div(overlap, float(accepted))
        f1 = _safe_div(2.0 * precision * recall, (precision + recall))

        return ProvisionalVsAcceptedSummaryRow(
            category=cat,
            accept_group=grp,
            percent=p,

            total=total,
            accepted=accepted,
            provisional=provisional,
            overlap=overlap,
            accepted_not_top=accepted_not_top,
            top_not_accepted=top_not_accepted,

            accepted_rate=accepted_rate,
            provisional_rate=provisional_rate,
            overlap_rate=overlap_rate,
            accepted_not_top_rate=accepted_not_top_rate,
            top_not_accepted_rate=top_not_accepted_rate,

            precision=precision,
            recall=recall,
            f1=f1,

            mean_overall=_safe_div(sum_overall, denom_total),
            mean_face=_safe_div(sum_face, denom_total),
            mean_comp=_safe_div(sum_comp, denom_total),
            mean_tech=_safe_div(sum_tech, denom_total),
        )

    out: List[ProvisionalVsAcceptedSummaryRow] = []

    # 1) ALL/ALL
    out.append(_summarize("ALL", "ALL", all_rows))

    # 2) category rollup: cat/ALL
    for cat in sorted(cat_buckets.keys()):
        out.append(_summarize(cat, "ALL", cat_buckets[cat]))

    # 3) group rows: cat/grp
    for (cat, grp), items in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        out.append(_summarize(cat, grp, items))

    meta = {
        "groups": len(buckets),
        "categories": len(cat_buckets),
        "total_rows": len(all_rows),
    }
    return out, meta


def write_provisional_vs_accepted_summary_csv(
    summary: List[ProvisionalVsAcceptedSummaryRow],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "category", "accept_group", "provisional_top_percent",
        "total", "accepted", "provisional", "overlap",
        "accepted_not_top", "top_not_accepted",

        "accepted_rate", "provisional_rate", "overlap_rate",
        "accepted_not_top_rate", "top_not_accepted_rate",

        "precision", "recall", "f1",

        "mean_overall", "mean_face", "mean_comp", "mean_tech",
    ]
    lines = [",".join(header) + "\n"]

    for r in summary:
        lines.append(
            ",".join(
                [
                    str(r.category),
                    str(r.accept_group),
                    f"{r.percent:.2f}",

                    str(r.total),
                    str(r.accepted),
                    str(r.provisional),
                    str(r.overlap),
                    str(r.accepted_not_top),
                    str(r.top_not_accepted),

                    f"{r.accepted_rate:.6f}",
                    f"{r.provisional_rate:.6f}",
                    f"{r.overlap_rate:.6f}",
                    f"{r.accepted_not_top_rate:.6f}",
                    f"{r.top_not_accepted_rate:.6f}",

                    f"{r.precision:.6f}",
                    f"{r.recall:.6f}",
                    f"{r.f1:.6f}",

                    f"{r.mean_overall:.4f}",
                    f"{r.mean_face:.4f}",
                    f"{r.mean_comp:.4f}",
                    f"{r.mean_tech:.4f}",
                ]
            )
            + "\n"
        )

    out_path.write_text("".join(lines), encoding="utf-8")
