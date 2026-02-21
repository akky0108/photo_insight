from __future__ import annotations

from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.brightness_comp_validate import discover_ranking_files


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("dummy", encoding="utf-8")


def test_discover_latest_only(tmp_path: Path) -> None:
    root = tmp_path / "output"

    _touch(root / "a" / "evaluation_ranking_2026-02-10.csv")
    _touch(root / "b" / "evaluation_ranking_2026-02-11.csv")

    files = discover_ranking_files(root_dir=root, date=None)

    assert len(files) == 1
    assert files[0].endswith("2026-02-11.csv")


def test_discover_specific_date(tmp_path: Path) -> None:
    root = tmp_path / "output"

    _touch(root / "x" / "evaluation_ranking_2026-02-09.csv")
    _touch(root / "y" / "evaluation_ranking_2026-02-11.csv")

    files = discover_ranking_files(root_dir=root, date="2026-02-09")

    assert len(files) == 1
    assert files[0].endswith("2026-02-09.csv")
