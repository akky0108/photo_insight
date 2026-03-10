from __future__ import annotations

import sys
from pathlib import Path

from photo_insight.image_loader import ImageLoader
from photo_insight.evaluators.portrait_quality.portrait_quality_evaluator import (
    PortraitQualityEvaluator,
)


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: python scripts/debug/debug_portrait_eval_init.py <nef_path>",
            flush=True,
        )
        return 2

    nef_path = Path(sys.argv[1]).expanduser().resolve()
    print(f"[START] evaluator init test: {nef_path}", flush=True)

    print("[1] before ImageLoader()", flush=True)
    loader = ImageLoader()
    print("[2] after ImageLoader()", flush=True)

    print("[3] before load_image", flush=True)
    image = loader.load_image(str(nef_path))
    print("[4] after load_image", flush=True)

    if image is None:
        print("[ERROR] image is None", flush=True)
        return 1

    print(f"[INFO] image.shape={getattr(image, 'shape', None)}", flush=True)
    print(f"[INFO] image.dtype={getattr(image, 'dtype', None)}", flush=True)

    print("[5] before PortraitQualityEvaluator(image_input=image)", flush=True)
    evaluator = PortraitQualityEvaluator(image_input=image)
    print("[6] after PortraitQualityEvaluator(image_input=image)", flush=True)

    print(repr(evaluator), flush=True)
    print("[END] evaluator init test ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())