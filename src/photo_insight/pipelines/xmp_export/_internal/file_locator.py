from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_csv(output_dir: Path, csv_glob: str, date: Optional[str]) -> Path:
    """
    date があるなら evaluation_ranking_{date}.csv を優先する。
    なければ csv_glob に一致する最新版を返す。
    """
    if date:
        candidate = output_dir / f"evaluation_ranking_{date}.csv"
        if candidate.exists():
            return candidate

    csv_files = sorted(
        output_dir.glob(csv_glob),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV found: {output_dir}/{csv_glob}")

    return csv_files[0]


def resolve_image_root(image_root: Path, date: Optional[str]) -> Path:
    """
    date が指定されたら /root/YYYY/YYYY-MM-DD に固定する。
    """
    if not date:
        return image_root

    year = date.split("-")[0]
    return image_root / year / date


def build_nef_index(base_dir: Path) -> dict[str, Path]:
    """
    base_dir 配下の .nef/.NEF を大小文字を区別せず index 化する。
    同名衝突時は最初に見つけたものを採用し、重複を警告する。
    """
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".nef":
            continue

        name = path.name
        if name in index:
            duplicates.setdefault(name, [index[name]]).append(path)
            continue
        index[name] = path

    if duplicates:
        sample = list(duplicates.items())[:5]
        print("⚠️ Duplicate NEF names detected under the search root. (showing up to 5)")
        for name, paths in sample:
            print(f"  - {name}:")
            for dup_path in paths:
                print(f"      {dup_path}")

    return index
