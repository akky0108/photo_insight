# tests/unit/tools/test_score_dist_validation.py
from __future__ import annotations

from src.tools.score_dist_validation import Criteria7015, validate_row_701_5


def _base_row():
    return {
        "type": "tech",
        "new_saturation_0plus1": 0.10,
        "current_accepted_ratio": 0.30,
        "new_accepted_ratio": 0.32,
        "direction_final": "higher_is_better",
        "new_discrete_ratio": 0.99,
        "new_in_range_ratio": 0.99,
        "current_tech_target_l1": 0.20,
        "new_tech_target_l1": 0.18,
    }


def test_validate_row_pass():
    row = _base_row()
    criteria = Criteria7015()
    ok, reasons = validate_row_701_5(row, criteria=criteria)
    assert ok is True
    assert reasons == []


def test_validate_row_fail_saturation():
    row = _base_row()
    row["new_saturation_0plus1"] = 0.50
    ok, reasons = validate_row_701_5(row, criteria=Criteria7015(sat_max=0.40))
    assert ok is False
    assert any("sat_0plus1" in r for r in reasons)


def test_validate_row_fail_accepted_delta():
    row = _base_row()
    row["new_accepted_ratio"] = 0.60
    ok, reasons = validate_row_701_5(
        row, criteria=Criteria7015(accepted_delta_max=0.10)
    )
    assert ok is False
    assert any("accepted_delta" in r for r in reasons)


def test_validate_row_fail_discrete_ratio():
    row = _base_row()
    row["new_discrete_ratio"] = 0.80
    ok, reasons = validate_row_701_5(row, criteria=Criteria7015(discrete_min=0.95))
    assert ok is False
    assert any("discrete_ratio" in r for r in reasons)


def test_validate_row_fail_in_range_ratio():
    row = _base_row()
    row["new_in_range_ratio"] = 0.50
    ok, reasons = validate_row_701_5(row, criteria=Criteria7015(in_range_min=0.98))
    assert ok is False
    assert any("in_range_ratio" in r for r in reasons)


def test_validate_row_direction_conflict_fail_when_enabled():
    row = _base_row()
    row["direction_final"] = "conflict"
    ok, reasons = validate_row_701_5(
        row, criteria=Criteria7015(direction_conflict_fail=True)
    )
    assert ok is False
    assert "direction_conflict" in reasons


def test_validate_row_require_target_l1_improve():
    row = _base_row()
    row["current_tech_target_l1"] = 0.10
    row["new_tech_target_l1"] = 0.20  # regressed
    ok, reasons = validate_row_701_5(
        row, criteria=Criteria7015(require_target_l1_improve=True)
    )
    assert ok is False
    assert any("tech_target_l1_regressed" in r for r in reasons)
