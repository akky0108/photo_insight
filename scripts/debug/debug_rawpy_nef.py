from __future__ import annotations

import sys
from pathlib import Path

import rawpy


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/debug/debug_rawpy_nef.py <nef_path>", flush=True)
        return 2

    nef_path = Path(sys.argv[1]).expanduser().resolve()
    print(f"[START] rawpy test: {nef_path}", flush=True)

    print("[1] before rawpy.imread", flush=True)
    with rawpy.imread(str(nef_path)) as raw:
        print("[2] after rawpy.imread", flush=True)

        print(f"raw_type={raw.raw_type}", flush=True)
        print(f"sizes={raw.sizes}", flush=True)
        print(f"color_desc={raw.color_desc!r}", flush=True)
        print(f"num_colors={raw.num_colors}", flush=True)

        print("[3] before postprocess", flush=True)
        rgb = raw.postprocess()
        print("[4] after postprocess", flush=True)

        print(f"rgb.shape={rgb.shape}", flush=True)
        print(f"rgb.dtype={rgb.dtype}", flush=True)

    print("[END] rawpy test ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())