from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from photo_insight.batch_processor.evaluation_rank.analysis.provisional_vs_accepted import (
    build_provisional_vs_accepted_summary,
    write_provisional_vs_accepted_summary_csv,
)


def _r(**kw: Any) -> Dict[str, Any]:
    base = {
        "category": "portrait",
        "accept_group": "a-1",
        "accepted_flag": 0,
        "provisional_top_percent_flag": 0,
        "provisional_top_percent": 10.0,
        "overall_score": 80.0,
        "score_face": 70.0,
        "score_composition": 60.0,
        "score_technical": 50.0,
    }
    base.update(kw)
    return base


def _approx(a: float, b: float, tol: float = 1e-9) -> None:
    assert abs(float(a) - float(b)) <= tol


def test_build_summary_counts_all_and_groups() -> None:
    rows: List[Dict[str, Any]] = [
        _r(file_name="a", accepted_flag=1, provisional_top_percent_flag=1),  # overlap
        _r(file_name="b", accepted_flag=1, provisional_top_percent_flag=0),  # accepted_not_top
        _r(file_name="c", accepted_flag=0, provisional_top_percent_flag=1),  # top_not_accepted
        _r(
            file_name="d",
            category="non_face",
            accept_group="non_face",
            accepted_flag=0,
            provisional_top_percent_flag=0,
        ),
    ]

    summary, meta = build_provisional_vs_accepted_summary(rows)
    assert meta["total_rows"] == 4
    assert meta["categories"] == 2  # portrait / non_face

    # --- ALL row (always first) ---
    all_row = summary[0]
    assert all_row.category == "ALL"
    assert all_row.accept_group == "ALL"
    assert all_row.total == 4
    assert all_row.accepted == 2
    assert all_row.provisional == 2
    assert all_row.overlap == 1
    assert all_row.accepted_not_top == 1
    assert all_row.top_not_accepted == 1

    # --- rates: denom = total (=4) ---
    _approx(all_row.accepted_rate, 2 / 4)
    _approx(all_row.provisional_rate, 2 / 4)
    _approx(all_row.overlap_rate, 1 / 4)
    _approx(all_row.accepted_not_top_rate, 1 / 4)
    _approx(all_row.top_not_accepted_rate, 1 / 4)

    # --- alignment metrics ---
    # precision = overlap / provisional = 1/2
    # recall    = overlap / accepted    = 1/2
    # f1        = 0.5
    _approx(all_row.precision, 1 / 2)
    _approx(all_row.recall, 1 / 2)
    _approx(all_row.f1, 0.5)

    # --- group rows exist ---
    keys = {(r.category, r.accept_group) for r in summary}
    assert ("portrait", "a-1") in keys
    assert ("non_face", "non_face") in keys

    # --- category rollup rows exist (Step2) ---
    assert ("portrait", "ALL") in keys
    assert ("non_face", "ALL") in keys


def test_write_csv(tmp_path: Path) -> None:
    rows = [_r(), _r(category="non_face", accept_group="non_face")]
    summary, _ = build_provisional_vs_accepted_summary(rows)

    out = tmp_path / "s.csv"
    write_provisional_vs_accepted_summary_csv(summary, out)
    text = out.read_text(encoding="utf-8")

    # header basics
    assert "category,accept_group,provisional_top_percent" in text

    # rates + alignment metrics are written
    assert "accepted_rate,provisional_rate,overlap_rate" in text
    assert "accepted_not_top_rate,top_not_accepted_rate" in text
    assert "precision,recall,f1" in text

    # ALL row exists
    assert "ALL,ALL" in text

    # Step2: category rollups are written
    assert "portrait,ALL" in text
    assert "non_face,ALL" in text
