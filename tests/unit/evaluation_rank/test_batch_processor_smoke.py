from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from photo_insight.batch_processor.evaluation_rank.contract import (
    INPUT_REQUIRED_COLUMNS,
    OUTPUT_COLUMNS,
)
from photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor import (
    EvaluationRankBatchProcessor,
)


def _write_min_input_csv(csv_path: Path, *, n_rows: int = 1) -> None:
    """
    INPUT_REQUIRED_COLUMNS を満たす最小CSVを作る。
    値は雑でOK（落ちないことが目的）。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for i in range(n_rows):
        # 雑な1行データ（列ごとに “それっぽい” ダミーを入れる）
        row: dict[str, str] = {c: "" for c in INPUT_REQUIRED_COLUMNS}

        # 必須の識別系
        row["file_name"] = f"dummy_{i:03d}.jpg"
        row["group_id"] = "A"
        row["subgroup_id"] = "1"
        row["shot_type"] = "upper_body"

        # bool系
        row["face_detected"] = "1"
        row["full_body_detected"] = "0"

        # faces / gaze は文字列として入ることが多いので JSON 文字列で
        row["faces"] = json.dumps([])
        row["gaze"] = json.dumps({"x": 0.0, "y": 0.0, "z": 0.0})

        # accepted_* は入力に必須
        row["accepted_flag"] = "0"
        row["accepted_reason"] = ""

        # --- score系 (0..1) ---
        score_like_keys = [
            # technical / face / composition scores
            "sharpness_score",
            "blurriness_score",
            "contrast_score",
            "noise_score",
            "local_sharpness_score",
            "local_contrast_score",
            "exposure_score",
            "face_sharpness_score",
            "face_contrast_score",
            "face_noise_score",
            "face_local_sharpness_score",
            "face_local_contrast_score",
            "face_exposure_score",
            "expression_score",
            "composition_rule_based_score",
            "face_position_score",
            "framing_score",
            "face_direction_score",
            "eye_contact_score",
            "lead_room_score",
            "body_composition_score",
            "composition_score",
            "rule_of_thirds_score",
            "face_composition_score",
            "face_blurriness_score",
            # brightness adjusted variants (rank sideで参照されうる)
            "blurriness_score_brightness_adjusted",
            "noise_score_brightness_adjusted",
            "face_blurriness_score_brightness_adjusted",
        ]
        for k in score_like_keys:
            if k in row:
                row[k] = "0.75"

        # --- raw/ratio/angle系 (適当な数値) ---
        raw_like_keys = [
            "sharpness_raw",
            "blurriness_raw",
            "contrast_raw",
            "noise_raw",
            "noise_sigma_midtone",
            "noise_sigma_used",
            "noise_mask_ratio",
            "local_sharpness_raw",
            "local_sharpness_std",
            "local_contrast_raw",
            "local_contrast_std",
            "mean_brightness",
            "face_sharpness_raw",
            "face_contrast_raw",
            "face_noise_raw",
            "face_noise_sigma_midtone",
            "face_noise_sigma_used",
            "face_noise_mask_ratio",
            "face_local_sharpness_std",
            "face_local_contrast_std",
            "face_mean_brightness",
            "yaw",
            "pitch",
            "roll",
            "delta_face_sharpness",
            "delta_face_contrast",
            "face_composition_raw",
            "face_blurriness_raw",
            "body_composition_raw",
            "composition_raw",
            "main_subject_center_x",
            "main_subject_center_y",
            "rule_of_thirds_raw",
            "headroom_ratio",
            "footroom_ratio",
            "side_margin_min_ratio",
            "full_body_cut_risk",
            "pose_score",
            "body_height_ratio",
            "body_center_y_ratio",
        ]
        for k in raw_like_keys:
            if k in row:
                row[k] = "1.0"

        # status / grade 系（最低限OKに寄せる）
        for k in list(row.keys()):
            if k.endswith("_eval_status") or k.endswith("_status"):
                row[k] = "ok"
        for k in list(row.keys()):
            if k.endswith("_grade"):
                row[k] = "good"

        rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INPUT_REQUIRED_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@pytest.mark.parametrize("date", ["2026-02-11"])
def test_evaluation_rank_batch_processor_smoke(tmp_path: Path, date: str) -> None:
    """
    最小CSVで processor.execute() が落ちず、
    出力CSVのヘッダが OUTPUT_COLUMNS と一致し、
    #706-1 provisional top% が実際に埋まることを確認する。
    """
    eval_dir = tmp_path / "temp"
    out_dir = tmp_path / "output"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # input CSV: 12行（10% -> ceil で2件）
    input_csv = eval_dir / f"evaluation_results_{date}.csv"
    _write_min_input_csv(input_csv, n_rows=12)

    # 最小config（paths を読む前提）+ #706-1 設定
    config_path = tmp_path / "evaluation_rank.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  evaluation_data_dir: {str(eval_dir)}",
                f"  output_data_dir: {str(out_dir)}",
                "",
                "evaluation_rank:",
                "  provisional_top_percent:",
                "    enabled: true",
                "    percent: 10",
            ]
        ),
        encoding="utf-8",
    )

    processor = EvaluationRankBatchProcessor(
        config_path=str(config_path),
        max_workers=1,
        date=date,
    )

    # setup が読む前提の SSOT をテスト側でも直指定（smokeの目的は I/O 経路の確認）
    processor.paths["evaluation_data_dir"] = str(eval_dir)
    processor.paths["output_data_dir"] = str(out_dir)

    processor.execute()

    output_csv = out_dir / f"evaluation_ranking_{date}.csv"
    assert output_csv.exists(), f"missing output: {output_csv}"

    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    # header contract
    assert header == OUTPUT_COLUMNS, "output header must match OUTPUT_COLUMNS (SSOT)"

    # #706-1: provisional values
    assert len(rows) == 12

    # percent is set for all rows
    percents = [float(r.get("provisional_top_percent") or 0.0) for r in rows]
    assert all(
        p == 10.0 for p in percents
    ), f"unexpected provisional_top_percent: {sorted(set(percents))}"

    # flag count uses ceil: 12 * 10% => 2
    flags = [int(float(r.get("provisional_top_percent_flag") or 0)) for r in rows]
    assert (
        sum(flags) == 2
    ), f"expected 2 top-percent rows, got sum={sum(flags)} flags={flags}"
