# tests/unit/tools/test_brightness_comp_validate.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

# --- make tools importable ---
import sys
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.brightness_comp_validate import (  # type: ignore
    run,
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    PAIRS_FILENAME,
)


def _write_ranking_csv(path: Path) -> None:
    """
    brightness_comp_validate が読むのは ranking CSV。
    必須列だけ用意して、最小で動くようにする。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "file_name",
        "mean_brightness",
        "blurriness_score",
        "blurriness_score_brightness_adjusted",
        "blurriness_raw",
        "noise_score",
        "noise_score_brightness_adjusted",
        "noise_raw",
    ]
    rows = [
        {
            "file_name": "a.jpg",
            "mean_brightness": "0.10",
            "blurriness_score": "0.75",
            "blurriness_score_brightness_adjusted": "0.70",
            "blurriness_raw": "0.0010",
            "noise_score": "0.50",
            "noise_score_brightness_adjusted": "0.55",
            "noise_raw": "-0.0012",
        },
        {
            "file_name": "b.jpg",
            "mean_brightness": "0.80",
            "blurriness_score": "0.25",
            "blurriness_score_brightness_adjusted": "0.30",
            "blurriness_raw": "0.0020",
            "noise_score": "0.75",
            "noise_score_brightness_adjusted": "0.70",
            "noise_raw": "-0.0009",
        },
    ]

    # 手書きCSV（csvモジュール使わずに依存減らす）
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_brightness_comp_validate_writes_ssot_outputs(tmp_path: Path) -> None:
    # arrange
    out_dir = tmp_path / "out"
    ranking_dir = tmp_path / "output" / "x"
    ranking_csv = ranking_dir / "evaluation_ranking_2026-02-11.csv"
    _write_ranking_csv(ranking_csv)

    # act（★ 新API）
    summary_path = run(
        root_dir=tmp_path / "output",
        date=None,  # 最新1件を自動選択
        out_dir=out_dir,
    )

    # assert: SSOT files exist
    assert summary_path.exists()

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == SCHEMA_VERSION
    assert "generated_at" in data

    assert data["inputs"]["n_files"] == 1
    assert data["inputs"]["n_rows"] == 2

    assert "metrics" in data
    assert "blurriness" in data["metrics"]
    assert "noise" in data["metrics"]

