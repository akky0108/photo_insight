# tests/unit/evaluation_rank/test_batch_processor_smoke.py
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from batch_processor.evaluation_rank.contract import (
    INPUT_REQUIRED_COLUMNS,
    OUTPUT_COLUMNS,
)

from batch_processor.evaluation_rank.evaluation_rank_batch_processor import (
    EvaluationRankBatchProcessor,
)
from tests.unit.processors.test_portrait_quality_batch_processor import processor


def _write_min_input_csv(csv_path: Path) -> None:
    """
    INPUT_REQUIRED_COLUMNS を満たす最小CSVを作る。
    値は雑でOK（落ちないことが目的）。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 雑な1行データ（列ごとに “それっぽい” ダミーを入れる）
    row = {}
    for c in INPUT_REQUIRED_COLUMNS:
        # まずは空で埋める（writer/normalizer側で落ちない設計ならこれでもOK）
        row[c] = ""

    # 重要そうなやつだけ最低限埋める（型事故・パース事故を避ける）
    row["file_name"] = "dummy.jpg"

    # スコア系: 0..1
    for k in [
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
        "face_blurriness_score",
        "face_composition_score",
    ]:
        if k in row:
            row[k] = "0.75"

    # raw系: 適当な数値
    for k in [
        "sharpness_raw",
        "blurriness_raw",
        "contrast_raw",
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
        "face_noise_sigma_midtone",
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
    ]:
        if k in row:
            row[k] = "1.0"

    # bool系
    row["face_detected"] = "1"
    row["full_body_detected"] = "0"

    # status / grade 系（最低限OKに寄せる）
    for k in list(row.keys()):
        if k.endswith("_eval_status") or k.endswith("_status"):
            row[k] = "ok"

    for k in list(row.keys()):
        if k.endswith("_grade"):
            row[k] = "good"

    # faces: 文字列として入る想定のことが多いので JSON 文字列で
    row["faces"] = json.dumps([])

    # gaze: dict文字列でもOK
    row["gaze"] = json.dumps({"x": 0.0, "y": 0.0, "z": 0.0})

    # group/subgroup/shot_type
    row["group_id"] = "A"
    row["subgroup_id"] = "1"
    row["shot_type"] = "upper_body"

    # contrib_* は INPUT_REQUIRED_COLUMNS に含まれているので埋めとく（空でも落ちないなら不要）
    for k in list(row.keys()):
        if k.startswith("contrib_"):
            row[k] = "0.0"

    # accepted_flag / accepted_reason は入力に必須なのでダミー
    row["accepted_flag"] = "0"
    row["accepted_reason"] = ""

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INPUT_REQUIRED_COLUMNS)
        w.writeheader()
        w.writerow(row)


@pytest.mark.parametrize("date", ["2026-02-11"])
def test_evaluation_rank_batch_processor_smoke(tmp_path: Path, date: str) -> None:
    """
    最小CSVで processor.execute() が落ちず、出力CSVのヘッダが OUTPUT_COLUMNS と一致すること。
    """
    eval_dir = tmp_path / "temp"
    out_dir = tmp_path / "output"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # input CSV
    input_csv = eval_dir / f"evaluation_results_{date}.csv"
    _write_min_input_csv(input_csv)

    # BaseBatchProcessor の config が paths を読む前提で最小configを書く
    # ※もし BaseBatchProcessor の期待スキーマが違う場合、ここだけ調整すればOK
    config_path = tmp_path / "evaluation_rank.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  evaluation_data_dir: {str(eval_dir)}",
                f"  output_data_dir: {str(out_dir)}",
            ]
        ),
        encoding="utf-8",
    )

    processor = EvaluationRankBatchProcessor(
        config_path=str(config_path),
        max_workers=1,
        date=date,
    )
    # BaseBatchProcessor が config を paths に反映する前に load_data が走る経路に備えて、
    # テスト側で SSOT を明示する（smokeの目的は I/O 経路の確認）
    processor.paths["evaluation_data_dir"] = str(eval_dir)
    processor.paths["output_data_dir"] = str(out_dir)

    processor.execute()

    # output ranking CSV
    output_csv = out_dir / f"evaluation_ranking_{date}.csv"
    assert output_csv.exists(), f"missing output: {output_csv}"

    with output_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

    assert header == OUTPUT_COLUMNS, "output header must match OUTPUT_COLUMNS (SSOT)"
