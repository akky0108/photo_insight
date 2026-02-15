# tests/unit/tools/test_brightness_comp_validate.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# --- make tools importable ---
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.brightness_comp_validate import (  # type: ignore
    run,
    SUMMARY_FILENAME,
    PAIRS_FILENAME,
)


def _write_ranking_csv(path: Path, *, n_rows: int = 40) -> None:
    """
    brightness_comp_validate は ranking CSV を読む。
    最小の必須列だけ用意して、数行以上（min_n対策）を生成する。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "file_name",
        "face_detected",
        "mean_brightness",
        "blurriness_score",
        "blurriness_score_brightness_adjusted",
        "noise_score",
        "noise_score_brightness_adjusted",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()

        # 明るさが低いほど少し救済される（delta=adj-before が負方向になりやすい）みたいな形を作る
        # （ただしここでは fail 判定の厳密性より “落ちない” を優先）
        for i in range(n_rows):
            b = 0.05 + (0.95 * (i / max(1, n_rows - 1)))  # 0.05..1.0
            before_blur = 0.75 if b < 0.5 else 0.25
            adj_blur = before_blur - 0.05 if b < 0.5 else before_blur + 0.05

            before_noise = 0.50 if b < 0.5 else 0.75
            adj_noise = before_noise + 0.05 if b < 0.5 else before_noise - 0.05

            w.writerow(
                {
                    "file_name": f"img_{i:03d}.jpg",
                    "face_detected": "1" if i % 2 == 0 else "0",
                    "mean_brightness": f"{b:.4f}",
                    "blurriness_score": f"{before_blur:.2f}",
                    "blurriness_score_brightness_adjusted": f"{adj_blur:.2f}",
                    "noise_score": f"{before_noise:.2f}",
                    "noise_score_brightness_adjusted": f"{adj_noise:.2f}",
                }
            )


def test_brightness_comp_validate_writes_ssot_outputs(tmp_path: Path) -> None:
    # arrange
    out_dir = tmp_path / "out"
    ranking_dir = tmp_path / "output" / "x"
    ranking_csv = ranking_dir / "evaluation_ranking_2026-02-11.csv"
    _write_ranking_csv(ranking_csv, n_rows=40)

    # act（★ 新API：ranking_glob 指定）
    summary_path = run(
        ranking_glob=str(tmp_path / "output" / "**" / "evaluation_ranking_*.csv"),
        out_dir=out_dir,
        emit_pairs_csv=False,
    )

    # assert: SSOT summary exists
    assert summary_path.exists()
    assert summary_path.name == SUMMARY_FILENAME

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # 入力ファイルが拾えていること
    assert "files" in data
    assert len(data["files"]) == 1

    # metrics があり、主要2指標が入っていること（最低限）
    assert "metrics" in data
    assert "blurriness" in data["metrics"]
    assert "noise" in data["metrics"]

    # result は pass / fail / skip のどれか
    assert data["result"] in ("pass", "fail", "skip")


def test_brightness_comp_validate_emits_pairs_csv(tmp_path: Path) -> None:
    # arrange
    out_dir = tmp_path / "out"
    ranking_dir = tmp_path / "output" / "x"
    ranking_csv = ranking_dir / "evaluation_ranking_2026-02-11.csv"
    _write_ranking_csv(ranking_csv, n_rows=35)

    # act
    summary_path = run(
        ranking_glob=str(tmp_path / "output" / "**" / "evaluation_ranking_*.csv"),
        out_dir=out_dir,
        emit_pairs_csv=True,
    )

    # assert
    assert summary_path.exists()
    pairs_path = out_dir / PAIRS_FILENAME
    assert pairs_path.exists()

    # pairs はヘッダだけでもOKだが、行があることを軽く確認
    text = pairs_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) >= 2  # header + 1 row


def test_brightness_comp_validate_no_files_returns_skip(tmp_path: Path) -> None:
    # arrange
    out_dir = tmp_path / "out"

    # act（存在しないglob）
    summary_path = run(
        ranking_glob=str(tmp_path / "output" / "**" / "evaluation_ranking_*.csv"),
        out_dir=out_dir,
        emit_pairs_csv=True,
    )

    # assert
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["result"] == "skip"
    assert "skip_reason" in data


def test_brightness_comp_validate_fail_on_raises_systemexit(tmp_path: Path) -> None:
    # arrange
    out_dir = tmp_path / "out"
    ranking_dir = tmp_path / "output" / "x"
    ranking_csv = ranking_dir / "evaluation_ranking_2026-02-11.csv"
    _write_ranking_csv(ranking_csv, n_rows=40)

    # 強制的に fail させるため、ルールを極端に厳しくする
    rules = {
        "min_n": 10,
        "skip_if_insufficient_n": False,
        "checks": {
            "corr_before_vs_adj_min": 0.9999,
            "p95_abs_delta_max": 0.00001,
            "corr_brightness_vs_delta_max": -0.9999,
        },
    }

    # act / assert
    with pytest.raises(SystemExit) as e:
        run(
            ranking_glob=str(tmp_path / "output" / "**" / "evaluation_ranking_*.csv"),
            out_dir=out_dir,
            emit_pairs_csv=False,
            rules=rules,
            fail_on=True,
        )
    assert e.value.code == 2
