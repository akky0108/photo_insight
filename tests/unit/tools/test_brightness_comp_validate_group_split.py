from __future__ import annotations

import csv
import json
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.brightness_comp_validate import run


def _write_ranking_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "file_name",
        "face_detected",
        "mean_brightness",
        "face_mean_brightness",
        "blurriness_score",
        "blurriness_score_brightness_adjusted",
        "blurriness_raw",
        "noise_score",
        "noise_score_brightness_adjusted",
        "noise_raw",
        "face_blurriness_score",
        "face_blurriness_score_brightness_adjusted",
        "face_blurriness_raw",
    ]

    # face 2行、non_face 1行（non_face は face_* を欠損にして n=0 になる想定）
    rows = [
        {
            "file_name": "face1.jpg",
            "face_detected": "1",
            "mean_brightness": "0.10",
            "face_mean_brightness": "0.12",
            "blurriness_score": "0.50",
            "blurriness_score_brightness_adjusted": "0.75",
            "blurriness_raw": "0.002",
            "noise_score": "0.50",
            "noise_score_brightness_adjusted": "0.60",
            "noise_raw": "-0.001",
            "face_blurriness_score": "0.50",
            "face_blurriness_score_brightness_adjusted": "0.80",
            "face_blurriness_raw": "0.0005",
        },
        {
            "file_name": "face2.jpg",
            "face_detected": "true",
            "mean_brightness": "0.20",
            "face_mean_brightness": "0.22",
            "blurriness_score": "0.50",
            "blurriness_score_brightness_adjusted": "0.70",
            "blurriness_raw": "0.003",
            "noise_score": "0.50",
            "noise_score_brightness_adjusted": "0.55",
            "noise_raw": "-0.0012",
            "face_blurriness_score": "0.50",
            "face_blurriness_score_brightness_adjusted": "0.78",
            "face_blurriness_raw": "0.0006",
        },
        {
            "file_name": "nonface.jpg",
            "face_detected": "0",
            "mean_brightness": "0.30",
            "face_mean_brightness": "",
            "blurriness_score": "0.50",
            "blurriness_score_brightness_adjusted": "0.65",
            "blurriness_raw": "0.004",
            "noise_score": "0.50",
            "noise_score_brightness_adjusted": "0.52",
            "noise_raw": "-0.0015",
            "face_blurriness_score": "",
            "face_blurriness_score_brightness_adjusted": "",
            "face_blurriness_raw": "",
        },
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_group_split_counts(tmp_path: Path) -> None:
    ranking_csv = tmp_path / "output" / "x" / "evaluation_ranking_2026-02-11.csv"
    _write_ranking_csv(ranking_csv)

    out_dir = tmp_path / "out"
    summary_path = run(root_dir=tmp_path / "output", date=None, out_dir=out_dir)
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert data["inputs"]["n_rows"] == 3
    assert data["inputs"]["n_rows_face"] == 2
    assert data["inputs"]["n_rows_non_face"] == 1

    # face_blurriness: face group は2件、non_face group は0件（欠損のため）
    fb = data["metrics"]["face_blurriness"]
    assert fb["face"]["n"] == 2
    assert fb["non_face"]["n"] == 0
