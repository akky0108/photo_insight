from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
import numpy as np


def ensure_gray255(gray: np.ndarray) -> np.ndarray:
    """
    グレースケール画像を float32 の 0..255 スケールに揃える。

    入力想定:
      - uint8: 0..255
      - uint16: 0..65535
      - float: 0..1 もしくは 0..255（混在対策で推定）

    返却:
      - float32, 0..255 目安（クリップ済み）
    """
    if not isinstance(gray, np.ndarray):
        raise TypeError("ensure_gray255 expects numpy.ndarray")

    x = gray.astype(np.float32)

    # uint16 相当（RAW後処理や 16bit グレイが来るケース）
    if gray.dtype == np.uint16:
        x = x / 65535.0 * 255.0
        return np.clip(x, 0.0, 255.0)

    # uint8 はそのまま 0..255
    if gray.dtype == np.uint8:
        return np.clip(x, 0.0, 255.0)

    # float / その他：推定
    # - max <= 1.5 なら 0..1 とみなす
    # - それ以外は 0..255 とみなす（ただし極端に大きい値はクリップ）
    mx = float(np.nanmax(x)) if x.size else 0.0
    if np.isfinite(mx) and mx <= 1.5:
        x = x * 255.0

    return np.clip(x, 0.0, 255.0)


def load_thresholds_sorted(
    config: Any,
    metric_key: str,
    defaults: Dict[str, float],
    names_in_order: Tuple[str, ...] | Iterable[str],
) -> Dict[str, float]:
    """
    config[metric_key]["discretize_thresholds_raw"] から閾値を読み、
    - defaults で補完
    - float変換
    - 単調増加になるようにソートして返す（事故防止）

    返却は {name: float, ...} で names_in_order と同じキーを必ず含む。
    """
    key = str(metric_key or "")

    # config の正規化
    cfg = config if isinstance(config, dict) else {}
    metric_cfg = cfg.get(key, {}) if isinstance(cfg, dict) else {}
    thr = (
        metric_cfg.get("discretize_thresholds_raw", {})
        if isinstance(metric_cfg, dict)
        else {}
    )
    if not isinstance(thr, dict):
        thr = {}

    names = tuple(names_in_order)

    # 値の取り出し（defaults fallback + float化）
    vals = []
    for n in names:
        v = thr.get(n, defaults.get(n))
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = float(defaults.get(n))
        vals.append(fv)

    # 単調性保証（ソート）
    vals_sorted = sorted(vals)

    out: Dict[str, float] = {}
    for n, v in zip(names, vals_sorted):
        out[n] = float(v)
    return out
