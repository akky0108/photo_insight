from __future__ import annotations


def test_normalize_row_parses_faces_and_bools(minimal_required_row):
    from photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor import _normalize_row_inplace

    row = dict(minimal_required_row)
    row["face_detected"] = "TRUE"
    row["full_body_detected"] = "0"

    _normalize_row_inplace(row)

    assert isinstance(row["faces"], list)
    assert row["face_detected"] is True
    assert row["full_body_detected"] is False
    assert "faces_parse_reason" in row
