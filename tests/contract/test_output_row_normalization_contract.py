from __future__ import annotations


def test_missing_columns_are_filled_with_empty_or_zero_by_contract():
    from photo_insight.batch_processor.evaluation_rank.writer import _normalize_row_for_output
    from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS

    out = _normalize_row_for_output({"file_name": "IMG_0001.NEF"}, OUTPUT_COLUMNS)

    assert set(out.keys()) == set(OUTPUT_COLUMNS)

    flag_keys = {"flag", "accepted_flag", "secondary_accept_flag"}

    for k in OUTPUT_COLUMNS:
        if k == "file_name":
            continue
        if k in flag_keys:
            assert out[k] == 0  # 欠損でもフラグは 0/1 正規化される契約
        else:
            assert out[k] == ""  # 欠損は "" 埋め


def test_none_values_are_converted_to_empty():
    from photo_insight.batch_processor.evaluation_rank.writer import _normalize_row_for_output
    from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS

    out = _normalize_row_for_output({"file_name": None, "accepted_reason": None}, OUTPUT_COLUMNS)
    assert out["file_name"] == ""
    assert out["accepted_reason"] == ""


def test_flags_are_normalized_to_int_0_or_1():
    from photo_insight.batch_processor.evaluation_rank.writer import _normalize_row_for_output
    from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS

    out = _normalize_row_for_output(
        {
            "file_name": "IMG_0001.NEF",
            "accepted_flag": "true",
            "secondary_accept_flag": "0",
            "flag": 5,
        },
        OUTPUT_COLUMNS,
    )

    assert out["accepted_flag"] in (0, 1)
    assert out["secondary_accept_flag"] in (0, 1)
    assert out["flag"] in (0, 1)


def test_filename_alias_is_accepted():
    from photo_insight.batch_processor.evaluation_rank.writer import _normalize_row_for_output
    from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS

    out = _normalize_row_for_output({"filename": "IMG_0001.NEF"}, OUTPUT_COLUMNS)
    assert out["file_name"] == "IMG_0001.NEF"


def test_extra_keys_are_not_emitted():
    from photo_insight.batch_processor.evaluation_rank.writer import _normalize_row_for_output
    from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS

    out = _normalize_row_for_output({"file_name": "IMG_0001.NEF", "hacker": "NOPE"}, OUTPUT_COLUMNS)
    assert "hacker" not in out
    assert set(out.keys()) == set(OUTPUT_COLUMNS)
