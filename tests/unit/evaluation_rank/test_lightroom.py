# tests/unit/evaluation_rank/test_lightroom.py
# -*- coding: utf-8 -*-

import pytest

from photo_insight.batch_processor.evaluation_rank import lightroom as lr


# -------------------------
# basic safety helpers
# -------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, False),
        ("", False),
        (True, True),
        (False, False),
        ("TRUE", True),
        ("False", False),
        ("1", True),
        ("0", False),
        ("yes", True),
        ("no", False),
        ("  y  ", True),
        ("  n  ", False),
        ("2", True),  # int(float("2")) => 2 => True
        ("0.0", False),
        ("1.0", True),
        ("garbage", False),
    ],
)
def test_safe_bool(value, expected):
    assert lr.safe_bool(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, 0),
        ("", 0),
        (True, 1),
        (False, 0),
        (1, 1),
        (0, 0),
        (2, 1),
        (0.0, 0),
        (1.0, 1),
        ("TRUE", 1),
        ("False", 0),
        ("yes", 1),
        ("no", 0),
        ("2", 1),
        ("0", 0),
        ("0.0", 0),
        ("1.0", 1),
        ("garbage", 0),
    ],
)
def test_safe_int_flag(value, expected):
    assert lr.safe_int_flag(value) == expected


@pytest.mark.parametrize(
    "v, expected",
    [
        (None, 0.0),
        ("", 0.0),
        ("garbage", 0.0),
        (-1, 0.0),
        (0, 0.0),
        (0.5, 50.0),
        (1.0, 100.0),
        (1.1, 1.1),  # already 0-100
        (70, 70.0),
        ("0.25", 25.0),
        ("75", 75.0),
    ],
)
def test_to_0_100(v, expected):
    assert lr.to_0_100(v) == expected


@pytest.mark.parametrize(
    "overall, expected",
    [
        (100, 5),
        (85, 5),
        (84.99, 4),
        (75, 4),
        (74.99, 3),
        (65, 3),
        (64.99, 2),
        (55, 2),
        (54.99, 1),
        (45, 1),
        (44.99, 0),
    ],
)
def test_score_to_rating(overall, expected):
    assert lr.score_to_rating(float(overall)) == expected


# -------------------------
# reason inference
# -------------------------


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("", 0),
        (None, 0),
        ("SEC:portrait group=A-1 ...", 1),
        ("SEC-RESCUE:xxx", 1),
        ("ACC-SEC-FILL:xxx", 1),
        ("portrait group=A-1 ...", 0),
    ],
)
def test_infer_secondary_from_reason(reason, expected):
    assert lr._infer_secondary_from_reason(reason or "") == expected


@pytest.mark.parametrize(
    "reason, expected",
    [
        ("", 0),
        ("SEC:portrait group=A-1 ...", 0),  # Green推定は別条件（"portrait " 先頭など）
        ("portrait group=A-1 st=face_only rank=1/31 o=80.82 ...", 1),
        ("non_face group=C-1 ...", 1),
        ("ACC-FILL:xxx", 1),
        ("ACC:xxx", 1),
        ("ACC-xxx", 1),
        ("rank=1/10 thr=70 overall=80", 1),  # 旧互換条件の想定
        ("random text", 0),
    ],
)
def test_infer_green_from_reason(reason, expected):
    assert lr._infer_green_from_reason(reason) == expected


# -------------------------
# shorten_reason_for_lr
# -------------------------


def test_shorten_reason_converts_tags_commas_to_slash_and_strips_spaces():
    reason = "portrait group=A-1 o=80.82 tags=eye_contact, framing, expression"
    out = lr.shorten_reason_for_lr(reason, max_len=200)
    assert "tags=eye_contact/framing/expression" in out
    assert " " not in out.split("tags=", 1)[1]  # tags以降はスペース除去される想定


def test_shorten_reason_replaces_score_labels_and_compacts_spaces():
    reason = "SEC: overall=80  score_face=90   score_composition=70 score_technical=95"
    out = lr.shorten_reason_for_lr(reason, max_len=200)
    assert "o=80" in out
    assert "f=90" in out
    assert "c=70" in out
    assert "t=95" in out
    assert "  " not in out


def test_shorten_reason_truncates():
    reason = "portrait " + ("x" * 500)
    out = lr.shorten_reason_for_lr(reason, max_len=90)
    assert len(out) <= 90
    assert out.startswith("portrait")


def test_shorten_reason_truncates_long_text():
    reason = "portrait group=A-1 " + ("x" * 500)
    out = lr.shorten_reason_for_lr(reason, max_len=90)
    assert len(out) <= 90
    assert out.startswith("portrait group=A-1")


def test_shorten_reason_keeps_tail_when_truncating():
    reason = "portrait group=A-1 " + ("x" * 500)
    out = lr.shorten_reason_for_lr(reason, max_len=200)
    assert len(out) <= 200
    assert "x" in out  # 末尾が消えてないこと


# -------------------------
# choose_color_label rules
# -------------------------


def test_choose_color_label_green_overrides_everything():
    assert (
        lr.choose_color_label(
            accepted_flag=1,
            secondary_flag=1,
            rating=0,
            top_flag=0,
            face_in_focus=False,
            eye_state="half",
        )
        == "Green"
    )


def test_choose_color_label_half_eye_is_red_when_not_accepted():
    assert (
        lr.choose_color_label(
            accepted_flag=0,
            secondary_flag=0,
            rating=5,
            top_flag=1,
            face_in_focus=True,
            eye_state="half",
        )
        == "Red"
    )


def test_choose_color_label_secondary_is_yellow():
    assert (
        lr.choose_color_label(
            accepted_flag=0,
            secondary_flag=1,
            rating=5,
            top_flag=1,
            face_in_focus=True,
            eye_state="",
        )
        == "Yellow"
    )


def test_choose_color_label_closed_eye_is_yellow_when_not_accepted():
    assert (
        lr.choose_color_label(
            accepted_flag=0,
            secondary_flag=0,
            rating=5,
            top_flag=1,
            face_in_focus=True,
            eye_state="closed",
        )
        == "Yellow"
    )


def test_choose_color_label_clear_fail_red_when_low_rating_and_not_in_focus():
    assert (
        lr.choose_color_label(
            accepted_flag=0,
            secondary_flag=0,
            rating=1,
            top_flag=1,
            face_in_focus=False,
            eye_state="",
        )
        == "Red"
    )


def test_choose_color_label_top_flag_blue_when_not_accepted_and_not_secondary():
    assert (
        lr.choose_color_label(
            accepted_flag=0,
            secondary_flag=0,
            rating=4,
            top_flag=1,
            face_in_focus=False,
            eye_state="",
        )
        == "Blue"
    )


def test_choose_color_label_face_in_focus_yellow_when_not_top():
    assert (
        lr.choose_color_label(
            accepted_flag=0,
            secondary_flag=0,
            rating=4,
            top_flag=0,
            face_in_focus=True,
            eye_state="",
        )
        == "Yellow"
    )


def test_choose_color_label_default_empty():
    assert (
        lr.choose_color_label(
            accepted_flag=0,
            secondary_flag=0,
            rating=4,
            top_flag=0,
            face_in_focus=False,
            eye_state="",
        )
        == ""
    )


# -------------------------
# apply_lightroom_fields integration
# -------------------------


def _base_row(**kwargs):
    row = {
        "overall_score": 80.82,
        "accepted_flag": 0,
        "secondary_accept_flag": 0,
        "flag": 0,
        "accepted_reason": "",
        "face_detected": True,
        "face_sharpness_score": 1.0,  # 0-1 想定（=100）
        "eye_state": "",
    }
    row.update(kwargs)
    return row


def test_apply_lightroom_fields_accepted_green_keywords_from_reason():
    r = _base_row(
        accepted_flag=1,
        accepted_reason="portrait group=A-1 o=80.82 tags=eye_contact, framing, expression",
    )
    lr.apply_lightroom_fields(r, keyword_max_len=200)

    assert r["lr_color_label"] == "Green"
    assert r["lr_rating"] == 4
    assert r["lr_labelcolor_key"] == "green"
    assert r["lr_label_display"] == "グリーン"
    # keywords should include tags normalized
    assert "tags=eye_contact/framing/expression" in r["lr_keywords"]


def test_apply_lightroom_fields_infer_green_from_reason_when_accepted_flag_missing():
    r = _base_row(
        accepted_flag=0,
        accepted_reason="portrait group=A-1 st=face_only rank=1/31 o=80.82 tags=eye_contact, framing, expression",
    )
    lr.apply_lightroom_fields(r, keyword_max_len=200)

    assert r["lr_color_label"] == "Green"
    assert "tags=eye_contact/framing/expression" in r["lr_keywords"]


def test_apply_lightroom_fields_infer_secondary_from_reason_sets_yellow_and_keywords():
    r = _base_row(
        accepted_flag=0,
        secondary_accept_flag=0,
        accepted_reason="SEC:portrait group=B-1 overall=73.69 tags=eye_contact, framing",
    )
    lr.apply_lightroom_fields(r, keyword_max_len=200)

    assert r["lr_color_label"] == "Yellow"
    assert "SEC:portrait" in r["lr_keywords"]  # shorten後も先頭は残る想定


def test_apply_lightroom_fields_blue_candidate_has_short_keywords():
    r = _base_row(
        accepted_flag=0,
        secondary_accept_flag=0,
        flag=1,  # top candidate
        overall_score=78.2,
        accepted_reason="",
        face_detected=False,
        face_sharpness_score=0.0,
    )
    lr.apply_lightroom_fields(r, keyword_max_len=200)

    assert r["lr_color_label"] == "Blue"
    assert r["lr_keywords"].startswith("CAND o=")
    assert "portrait" not in r["lr_keywords"]


def test_apply_lightroom_fields_half_eye_forces_red_when_not_accepted():
    r = _base_row(
        accepted_flag=0,
        secondary_accept_flag=0,
        flag=1,
        overall_score=90,
        eye_state="half",
    )
    lr.apply_lightroom_fields(r, keyword_max_len=200)

    assert r["lr_color_label"] == "Red"
    assert r["lr_keywords"] == ""


def test_apply_lightroom_fields_face_in_focus_yellow_when_not_top_and_not_secondary():
    # Not accepted, not secondary, not top. If face_in_focus => Yellow
    r = _base_row(
        accepted_flag=0,
        secondary_accept_flag=0,
        flag=0,
        overall_score=60,  # rating=2
        face_detected=True,
        face_sharpness_score=1.0,  # 100 => focus True
        accepted_reason="",
    )
    lr.apply_lightroom_fields(r, keyword_max_len=200)

    assert r["lr_color_label"] == "Yellow"
    assert (
        r["lr_keywords"] == ""
    )  # Yellow でも effective_secondary/accepted じゃないので keywords は空の設計
