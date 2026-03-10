from __future__ import annotations

import sys
from pathlib import Path

from photo_insight.image_loader import ImageLoader


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: python scripts/debug/debug_image_loader_nef.py <nef_path>",
            flush=True,
        )
        return 2

    nef_path = Path(sys.argv[1]).expanduser().resolve()
    print(f"[START] image_loader test: {nef_path}", flush=True)

    print("[1] before ImageLoader()", flush=True)
    loader = ImageLoader()
    print("[2] after ImageLoader()", flush=True)

    print("[3] before loader.load_image", flush=True)
    image = loader.load_image(str(nef_path))
    print("[4] after loader.load_image", flush=True)

    if image is None:
        print("[INFO] image is None", flush=True)
    else:
        print(f"[INFO] image.shape={getattr(image, 'shape', None)}", flush=True)
        print(f"[INFO] image.dtype={getattr(image, 'dtype', None)}", flush=True)

    print("[END] image_loader test ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())