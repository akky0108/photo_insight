from __future__ import annotations

import sys
from pathlib import Path

import rawpy


def dump(path: Path) -> None:
    print("=" * 80, flush=True)
    print(path, flush=True)

    print("[1] before rawpy.imread", flush=True)
    with rawpy.imread(str(path)) as raw:
        print("[2] after rawpy.imread", flush=True)
        print(f"raw_type={raw.raw_type}", flush=True)
        print(f"num_colors={raw.num_colors}", flush=True)
        print(f"color_desc={raw.color_desc!r}", flush=True)
        print(f"sizes={raw.sizes}", flush=True)
        print(
            f"black_level_per_channel={getattr(raw, 'black_level_per_channel', None)}",
            flush=True,
        )
        print(
            f"camera_whitebalance={getattr(raw, 'camera_whitebalance', None)}",
            flush=True,
        )
        print(
            f"daylight_whitebalance={getattr(raw, 'daylight_whitebalance', None)}",
            flush=True,
        )


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: python scripts/debug/compare_rawpy_meta.py <nef1> <nef2>",
            flush=True,
        )
        return 2

    for arg in sys.argv[1:]:
        dump(Path(arg).expanduser().resolve())

    print("[END] compare_rawpy_meta ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())