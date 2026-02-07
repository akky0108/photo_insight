from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def ensure_gray255(gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale image to float32 in [0, 255] scale.

    Accepts:
      - uint8: 0..255
      - uint16: 0..65535
      - float: 0..1 or 0..255 (heuristic by max)

    This is the critical part to keep Contrast thresholds consistent.
    """
    if not isinstance(gray, np.ndarray):
        raise TypeError("gray_not_ndarray")
    if gray.size == 0:
        raise ValueError("gray_empty")

    if gray.dtype == np.uint8:
        out = gray.astype(np.float32)
    elif gray.dtype == np.uint16:
        out = gray.astype(np.float32) / 65535.0 * 255.0
    else:
        f = gray.astype(np.float32)
        mx = float(np.nanmax(f)) if f.size else 0.0
        # if it looks like 0..1 floats, scale up
        if mx <= 1.5:
            out = f * 255.0
        else:
            out = f

    return np.clip(out, 0.0, 255.0)


def load_thresholds_sorted(
    config: Mapping[str, Any] | None,
    *,
    metric_key: str,
    defaults: Mapping[str, float],
    names_in_order: tuple[str, ...],
) -> dict[str, float]:
    """
    Read config[metric_key]['discretize_thresholds_raw'] with fallback, and enforce monotone increasing.
    """
    cfg = config or {}
    metric_cfg = cfg.get(metric_key, {}) if isinstance(cfg, dict) else {}
    thr = metric_cfg.get("discretize_thresholds_raw", {}) if isinstance(metric_cfg, dict) else {}
    if not isinstance(thr, dict):
        thr = {}

    vals = []
    for k in names_in_order:
        v = thr.get(k, defaults[k])
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(float(defaults[k]))

    vals = sorted(vals)
    return {k: float(v) for k, v in zip(names_in_order, vals)}
