import math
import pytest

from photo_insight.batch_processor.evaluation_rank.acceptance import AcceptanceEngine, AcceptRules


def _row(
    file_name: str,
    *,
    group_id: str,
    subgroup_id: str,
    shot_type: str = "upper_body",
    face_detected: bool = True,
    overall: float = 70.0,
    face: float = 70.0,
    comp: float = 60.0,
    tech: float = 95.0,
    # eye policy fields (optional)
    eye_prob=None,
    eye_sz=None,
):
    r = {
        "file_name": file_name,
        "group_id": group_id,
        "subgroup_id": subgroup_id,
        "shot_type": shot_type,
        "face_detected": face_detected,

        "overall_score": overall,
        "score_face": face,
        "score_composition": comp,
        "score_technical": tech,
    }
    if eye_prob is not None:
        r["eye_closed_prob_best"] = eye_prob
    if eye_sz is not None:
        r["eye_patch_size_best"] = eye_sz
    return r


def _count_green(rows):
    return sum(1 for r in rows if int(r.get("accepted_flag", 0)) == 1)


def _count_green_by_group(rows):
    out = {}
    for r in rows:
        g = str(r.get("accept_group") or "")
        out[g] = out.get(g, 0) + (1 if int(r.get("accepted_flag", 0)) == 1 else 0)
    return out


def test_green_total_global_is_preserved():
    # total=150 => global ratio_large 0.20 => 30
    rules = AcceptRules(green_per_group_enabled=True)
    eng = AcceptanceEngine(rules)

    rows = []
    for i in range(75):
        rows.append(_row(f"A_{i}.NEF", group_id="A", subgroup_id="1", overall=80 - i * 0.01))
    for i in range(75):
        rows.append(_row(f"B_{i}.NEF", group_id="B", subgroup_id="1", overall=79 - i * 0.01))

    eng.apply_accepted_flags(rows)
    assert _count_green(rows) == 30


def test_green_is_not_all_taken_by_one_group():
    # A-1 is very strong, B-1 weaker, but still should receive some greens (min_each=1).
    rules = AcceptRules(green_per_group_enabled=True, green_per_group_min_each=1)
    eng = AcceptanceEngine(rules)

    rows = []
    # A-1: 120 strong
    for i in range(120):
        rows.append(_row(f"A_{i}.NEF", group_id="A", subgroup_id="1", overall=90 - i * 0.01, face=80, comp=60))
    # B-1: 30 weak-ish
    for i in range(30):
        rows.append(_row(f"B_{i}.NEF", group_id="B", subgroup_id="1", overall=70 - i * 0.01, face=75, comp=58))

    eng.apply_accepted_flags(rows)
    by = _count_green_by_group(rows)

    assert _count_green(rows) == 30  # global preserved
    assert by.get("A-1", 0) > 0
    assert by.get("B-1", 0) > 0  # at least 1


def test_when_global_smaller_than_groups_pick_strong_groups_first():
    # Force global green_total=2 while having 4 groups
    rules = AcceptRules(
        green_per_group_enabled=True,
        green_ratio_small=0.01,   # make global tiny
        green_ratio_mid=0.01,
        green_ratio_large=0.01,
        green_min_total=0,
        green_per_group_min_each=1,
    )
    eng = AcceptanceEngine(rules)

    rows = []
    # 4 groups, each 1 row
    rows.append(_row("G1.NEF", group_id="A", subgroup_id="1", overall=90))
    rows.append(_row("G2.NEF", group_id="A", subgroup_id="2", overall=80))
    rows.append(_row("G3.NEF", group_id="B", subgroup_id="1", overall=70))
    rows.append(_row("G4.NEF", group_id="B", subgroup_id="2", overall=60))

    # total_n=4, ratio_small=0.01 => ceil(0.04)=1 (but min_total=0 => 1)
    # We want global=2, so bump via green_ratio_small:
    # easiest: override by making 0.4 => ceil(1.6)=2
    rules2 = rules.__class__(**{**rules.__dict__, "green_ratio_small": 0.4})
    eng = AcceptanceEngine(rules2)

    eng.apply_accepted_flags(rows)
    greens = [r["file_name"] for r in rows if int(r.get("accepted_flag", 0)) == 1]
    assert len(greens) == 2
    assert "G1.NEF" in greens
    assert "G2.NEF" in greens


def test_eye_half_is_not_selected_as_green():
    rules = AcceptRules(green_per_group_enabled=True)
    eng = AcceptanceEngine(rules)

    rows = []
    # Group A-1 has 3 candidates; top one is half-eye -> should be skipped
    rows.append(_row("A_bad.NEF", group_id="A", subgroup_id="1", overall=95, eye_prob=0.90, eye_sz=120))  # half range
    rows.append(_row("A_ok1.NEF", group_id="A", subgroup_id="1", overall=94, eye_prob=0.10, eye_sz=120))
    rows.append(_row("A_ok2.NEF", group_id="A", subgroup_id="1", overall=93, eye_prob=0.10, eye_sz=120))

    # Keep global small to see behavior clearly
    rules2 = rules.__class__(**{**rules.__dict__, "green_ratio_small": 0.67, "green_min_total": 0})
    eng = AcceptanceEngine(rules2)

    eng.apply_accepted_flags(rows)
    greens = [r["file_name"] for r in rows if int(r.get("accepted_flag", 0)) == 1]
    assert "A_bad.NEF" not in greens
    assert len(greens) == 2


def test_backfill_fills_when_group_quota_cannot_be_met_by_content_gate():
    # Group A has only content-gate-failing rows; B has good rows.
    rules = AcceptRules(green_per_group_enabled=True, green_per_group_min_each=1, green_min_total=0)
    eng = AcceptanceEngine(rules)

    rows = []
    # A-1 fails content gate: face low & comp low
    for i in range(10):
        rows.append(_row(f"A_{i}.NEF", group_id="A", subgroup_id="1", overall=90 - i, face=40, comp=40))
    # B-1 passes
    for i in range(10):
        rows.append(_row(f"B_{i}.NEF", group_id="B", subgroup_id="1", overall=80 - i, face=75, comp=58))

    # total=20 => ratio_small 0.30 => 6
    eng.apply_accepted_flags(rows)
    assert _count_green(rows) == 6

    by = _count_green_by_group(rows)
    # A-1 は gate で埋められないので、B で backfill されるはず
    assert by.get("B-1", 0) >= 6
