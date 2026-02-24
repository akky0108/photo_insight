# tests/batch_processor/evaluation_rank/test_acceptance_green_quota_backfill.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from photo_insight.batch_processor.evaluation_rank.acceptance import (
    AcceptanceEngine,
    AcceptRules,
)

Row = Dict[str, Any]


def _make_row(
    i: int,
    *,
    overall: float = 80.0,
    face: float = 80.0,
    comp: float = 80.0,
    tech: float = 100.0,
    face_detected: bool = True,
    shot_type: str = "upper_body",
    group_id: str = "A",
    subgroup_id: str = "1",
) -> Row:
    """
    AcceptanceEngine が参照するキーだけ入れる最小 Row。
    contract の全カラムは不要。
    """
    return {
        "file_name": f"IMG_{i:04d}.NEF",
        "group_id": group_id,
        "subgroup_id": subgroup_id,
        "shot_type": shot_type,
        "face_detected": face_detected,
        "overall_score": overall,
        "score_face": face,
        "score_composition": comp,
        "score_technical": tech,
        # build_reason_pro が参照する系（なくても safe_float で 0 になるが、タグ検証しやすいように入れておく）
        "eye_contact_score": 1.0,
        "face_direction_score": 1.0,
        "framing_score": 1.0,
        "face_position_score": 1.0,
        "composition_rule_based_score": 1.0,
        "rule_of_thirds_score": 1.0,
        "lead_room_score": 1.0,
        "body_composition_score": 1.0,
        "expression_score": 1.0,
        "debug_expr_effective": 1.0,
        "debug_half_penalty": 0.0,
    }


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, 0.30),
        (60, 0.30),
        (61, 0.25),
        (120, 0.25),
        (121, 0.20),
        (150, 0.20),
    ],
)
def test_green_ratio_by_count_boundaries(n: int, expected: float) -> None:
    engine = AcceptanceEngine(AcceptRules())
    assert engine._green_ratio_by_count(n) == pytest.approx(expected)


def test_green_total_is_ceiled_and_min_applied() -> None:
    # green_min_total が効くケース
    rules = AcceptRules(
        green_ratio_small=0.01,
        green_ratio_mid=0.01,
        green_ratio_large=0.01,
        green_min_total=3,
        green_count_small_max=60,
        green_count_mid_max=120,
    )
    engine = AcceptanceEngine(rules)

    rows = [_make_row(i, overall=80.0) for i in range(10)]  # 10枚 * 1% = 0.1 → ceil=1 だが min=3
    thresholds = engine.apply_accepted_flags(rows)

    greens = [r for r in rows if int(r.get("accepted_flag", 0)) == 1]
    assert len(greens) == 3
    assert thresholds["green_total"] == pytest.approx(3.0)


def test_backfill_fills_quota_when_strict_gate_insufficient() -> None:
    """
    strict gate を意図的に通らない行を多数作り、
    relaxed/forced backfill で green_total を満たすことを確認する。
    """
    rules = AcceptRules(
        green_ratio_large=0.20,
        green_count_small_max=60,
        green_count_mid_max=120,
    )
    engine = AcceptanceEngine(rules)

    n = 150
    expected_green_total = int(math.ceil(n * 0.20))  # 30

    rows: List[Row] = []

    # 上位 10件だけ strict gate を通す（upper_body: face>=70 & comp>=55）
    for i in range(10):
        rows.append(_make_row(i, overall=100.0 - i, face=75.0, comp=60.0))

    # 残りは strict gate を落とす（face>=70 だけ満たすが comp を 49 にして落とす）
    # relaxed gate（face>=68 & comp>=50）も落とす or 通すを混ぜる
    # → 最終 forced backfill まで使っても OK
    for i in range(10, n):
        # overall は上位ほど高いが、gate で落ちるようにする
        rows.append(_make_row(i, overall=100.0 - i, face=72.0, comp=49.0))

    thresholds = engine.apply_accepted_flags(rows)

    greens = [r for r in rows if int(r.get("accepted_flag", 0)) == 1]
    assert thresholds["green_total"] == pytest.approx(float(expected_green_total))
    assert len(greens) == expected_green_total

    # backfill 由来が混ざっているはず（suffix 入れてる実装前提）
    # strict 通過分は suffix無し、埋めは FILL_* が付く
    reasons = [str(r.get("accepted_reason", "")) for r in greens]
    assert any("FILL_" in x for x in reasons)


def test_green_promote_clears_secondary_flag() -> None:
    """
    Green にした場合 secondary_accept_flag は必ず 0 になる（混在防止）。
    """
    rules = AcceptRules(green_ratio_small=0.50, green_count_small_max=60, green_count_mid_max=120)
    engine = AcceptanceEngine(rules)

    rows = [_make_row(i, overall=80.0 - i * 0.1, face=80.0, comp=80.0) for i in range(20)]
    thresholds = engine.apply_accepted_flags(rows)

    green_total = int(thresholds["green_total"])
    greens = [r for r in rows if int(r.get("accepted_flag", 0)) == 1]
    assert len(greens) == green_total

    for r in greens:
        assert int(r.get("secondary_accept_flag", 0)) == 0
        assert str(r.get("secondary_accept_reason", "")) == ""


def test_eye_state_policy_half_forces_reject() -> None:
    """
    half_min <= prob < closed_min なら accepted_flag/secondary_accept_flag を 0 に落とす。
    """
    engine = AcceptanceEngine(AcceptRules(green_ratio_small=0.50))

    rows = [_make_row(i) for i in range(10)]
    # 1件だけ half 条件にする
    rows[0]["eye_closed_prob_best"] = 0.90
    rows[0]["eye_patch_size_best"] = 100

    engine.apply_accepted_flags(rows)

    assert int(rows[0].get("accepted_flag", 0)) == 0
    assert int(rows[0].get("secondary_accept_flag", 0)) == 0
    assert "EYE_HALF_NG" in str(rows[0].get("accepted_reason", ""))
