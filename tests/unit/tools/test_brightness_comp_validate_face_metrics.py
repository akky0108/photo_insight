from __future__ import annotations

import csv
import json
from pathlib import Path

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.brightness_comp_validate import run, SCHEMA_VERSION


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

    # 2行、face_detected=True で face_* が有効
    rows = [
        {
            "file_name": "a.jpg",
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
            "file_name": "b.jpg",
            "face_detected": "1",
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
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_face_metrics_present(tmp_path: Path) -> None:
    ranking_csv = tmp_path / "output" / "x" / "evaluation_ranking_2026-02-11.csv"
    _write_ranking_csv(ranking_csv)

    out_dir = tmp_path / "out"
    summary_path = run(root_dir=tmp_path / "output", date=None, out_dir=out_dir)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == SCHEMA_VERSION

    assert "face_blurriness" in data["metrics"]
    assert "all" in data["metrics"]["face_blurriness"]
    assert data["metrics"]["face_blurriness"]["all"]["n"] == 2
