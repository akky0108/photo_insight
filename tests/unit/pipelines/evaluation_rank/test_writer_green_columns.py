from __future__ import annotations

import csv

from photo_insight.pipelines.evaluation_rank.writer import write_ranking_csv


def test_write_ranking_csv_includes_green_columns(tmp_path):
    output_csv = tmp_path / "ranking.csv"

    rows = [
        {
            "file_name": "a.NEF",
            "group_id": "A",
            "subgroup_id": "1",
            "shot_type": "face_only",
            "face_detected": True,
            "category": "portrait",
            "overall_score": 0.88,
            "flag": 0,
            "accepted_flag": 1,
            "secondary_accept_flag": 0,
            "is_green": True,
            "green_minimum_pass": True,
            "green_keep_reasons": "sns_expression|model_face_exposure",
            "green_reject_reasons": "",
            "green_decision_version": "v2-real",
            "provisional_top_percent_flag": 0,
            "provisional_top_percent": 0,
            "accepted_reason": "accepted",
        }
    ]

    columns = write_ranking_csv(
        output_csv=output_csv,
        rows=rows,
        sort_for_ranking=False,
    )

    assert "is_green" in columns
    assert "green_minimum_pass" in columns
    assert "green_keep_reasons" in columns
    assert "green_reject_reasons" in columns
    assert "green_decision_version" in columns

    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    assert row["is_green"] == "1"
    assert row["green_minimum_pass"] == "1"
    assert row["green_keep_reasons"] == "sns_expression|model_face_exposure"
    assert row["green_decision_version"] == "v2-real"
