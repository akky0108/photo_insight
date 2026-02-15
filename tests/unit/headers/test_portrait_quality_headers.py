# tests/unit/headers/test_portrait_quality_headers.py
from photo_insight.portrait_quality_header import PortraitQualityHeaderGenerator

def test_all_headers_include_acceptance_and_deltas():
    headers = PortraitQualityHeaderGenerator().get_all_headers()

    expected_keys = [
        "lead_room_score",
        "delta_face_sharpness",
        "delta_face_contrast",
        "accepted_flag",
        "accepted_reason",
    ]

    for key in expected_keys:
        assert key in headers, f"{key} is missing from CSV headers"

def test_save_results_writes_acceptance_columns(tmp_path):
    from photo_insight.portrait_quality_batch_processor import PortraitQualityBatchProcessor

    p = PortraitQualityBatchProcessor(config_path=None, logger=None)
    out = tmp_path / "out.csv"

    row = {
        "file_name": "a.jpg",
        "accepted_flag": True,
        "accepted_reason": "face_quality",
        "lead_room_score": 0.2,
        "delta_face_sharpness": 5.0,
        "delta_face_contrast": 3.0,
    }

    p.save_results([row], out)

    text = out.read_text()
    assert "accepted_flag" in text
    assert "accepted_reason" in text
    assert "face_quality" in text


def test_headers_include_full_body_columns():
    headers = PortraitQualityHeaderGenerator().get_all_headers()

    expected_keys = [
        "full_body_detected",
        "pose_score",
        "headroom_ratio",
        "footroom_ratio",
        "side_margin_min_ratio",
        "full_body_cut_risk",
    ]

    for key in expected_keys:
        assert key in headers, f"{key} is missing from CSV headers"
