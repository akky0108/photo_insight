# tests/unit/evaluation_rank/test_provisional_top_percent.py
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from photo_insight.batch_processor.evaluation_rank.provisional import apply_provisional_top_percent


def _mk_records(scores: List[Any]) -> List[Dict[str, Any]]:
    """
    score_key=overall_score を前提に records を作る。
    追加キーは不要（provisional は score しか見ない）。
    """
    return [{"file_name": f"f{i:03d}.jpg", "overall_score": s} for i, s in enumerate(scores)]


def _count_flag(records: List[Dict[str, Any]]) -> int:
    return sum(int(r.get("provisional_top_percent_flag") or 0) for r in records)


def test_basic_10pct_of_10_is_1() -> None:
    recs = _mk_records([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    apply_provisional_top_percent(recs, percent=10)

    assert len(recs) == 10
    assert _count_flag(recs) == 1
    assert all(float(r["provisional_top_percent"]) == 10.0 for r in recs)


def test_ceil_works_12pct10_is_2() -> None:
    # 12件 × 10% = 1.2 -> ceil 2
    recs = _mk_records([float(i) for i in range(12)])
    apply_provisional_top_percent(recs, percent=10)

    assert len(recs) == 12
    assert _count_flag(recs) == 2


def test_percent_0_is_all_zero() -> None:
    recs = _mk_records([0.5, 0.6, 0.7])
    apply_provisional_top_percent(recs, percent=0)

    assert _count_flag(recs) == 0
    assert all(float(r["provisional_top_percent"]) == 0.0 for r in recs)


def test_percent_100_is_all_one() -> None:
    recs = _mk_records([0.5, 0.6, 0.7, 0.8])
    apply_provisional_top_percent(recs, percent=100)

    assert _count_flag(recs) == 4
    assert all(float(r["provisional_top_percent"]) == 100.0 for r in recs)


def test_order_is_not_modified() -> None:
    recs = _mk_records([0.1, 0.9, 0.2, 0.8])
    before = [r["file_name"] for r in recs]

    apply_provisional_top_percent(recs, percent=50)

    after = [r["file_name"] for r in recs]
    assert after == before


@pytest.mark.parametrize("percent", [None, "10", "10.0", "garbage", -10, 200])
def test_percent_is_sanitized(percent: Any) -> None:
    recs = _mk_records([0.1, 0.2, 0.3, 0.4])
    apply_provisional_top_percent(recs, percent=percent)

    # always set percent_key for all rows
    assert "provisional_top_percent" in recs[0]
    p = float(recs[0]["provisional_top_percent"])
    assert 0.0 <= p <= 100.0


def test_score_none_or_unparseable_is_bottom() -> None:
    # None / unparseable は -inf 扱いで bottom に落ちる想定
    recs = _mk_records([0.9, None, "abc", 0.8])
    apply_provisional_top_percent(recs, percent=25)  # 4件の25% => 1件

    # topは 0.9 のはず（index 0）
    assert _count_flag(recs) == 1
    assert recs[0]["provisional_top_percent_flag"] == 1
    assert recs[1]["provisional_top_percent_flag"] == 0
    assert recs[2]["provisional_top_percent_flag"] == 0
    assert recs[3]["provisional_top_percent_flag"] == 0
