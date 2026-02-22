# tests/unit/evaluation_rank/test_provisional_top_percent.py
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from photo_insight.batch_processor.evaluation_rank.provisional import (
    apply_provisional_top_percent,
)


def _mk_records(scores: List[Any]) -> List[Dict[str, Any]]:
    """
    score_key=overall_score を前提に records を作る。
    provisional は score しか見ない（他キーは任意）。
    """
    return [{"file_name": f"f{i:03d}.jpg", "overall_score": s} for i, s in enumerate(scores)]


def _i01(v: Any) -> int:
    """bool/int/float/str を 0/1 に寄せる。落ちないのが最優先。"""
    if isinstance(v, bool):
        return 1 if v else 0
    if v is None:
        return 0
    try:
        return 1 if int(float(v)) != 0 else 0
    except Exception:
        s = str(v).strip().lower()
        return 1 if s in ("1", "true", "t", "yes", "y") else 0


def _count_flag(records: List[Dict[str, Any]]) -> int:
    return sum(_i01(r.get("provisional_top_percent_flag")) for r in records)


def _flagged_files(records: List[Dict[str, Any]]) -> List[str]:
    return [r["file_name"] for r in records if _i01(r.get("provisional_top_percent_flag")) == 1]


def test_basic_10pct_of_10_is_1() -> None:
    recs = _mk_records([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    apply_provisional_top_percent(recs, percent=10)

    assert len(recs) == 10
    assert _count_flag(recs) == 1
    assert all(float(r["provisional_top_percent"]) == 10.0 for r in recs)


def test_ceil_works_12_items_10pct_is_2() -> None:
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
    assert _i01(recs[0]["provisional_top_percent_flag"]) == 1
    assert _i01(recs[1]["provisional_top_percent_flag"]) == 0
    assert _i01(recs[2]["provisional_top_percent_flag"]) == 0
    assert _i01(recs[3]["provisional_top_percent_flag"]) == 0


def test_tie_is_index_based_topk() -> None:
    """
    仕様: 同点は index 方式（ソート後の上位k件をそのまま取る）
    → 同点が並んだとき、先に出現したものが優先される。
    """
    # 0.9 が同点で2件、k=1なので先の f000 が選ばれるはず
    recs = _mk_records([0.9, 0.9, 0.1, 0.2])
    apply_provisional_top_percent(recs, percent=25)  # 4件の25% => 1件

    assert _count_flag(recs) == 1
    assert _flagged_files(recs) == ["f000.jpg"]


def test_independent_from_accepted_flag() -> None:
    """
    provisional は accepted 判定とは独立であること。
    （accepted が 1 でも top% に入らなければ provisional は 0）
    """
    recs = _mk_records([0.1, 0.2, 0.3, 0.4])
    # accepted を適当に付与（provisional 側は見ない想定）
    recs[0]["accepted_flag"] = 1  # 最下位
    recs[3]["accepted_flag"] = 0  # 最上位

    apply_provisional_top_percent(recs, percent=25)  # 4件の25% => 1件
    assert _flagged_files(recs) == ["f003.jpg"]
    assert _i01(recs[0].get("provisional_top_percent_flag")) == 0
