from __future__ import annotations

from typing import Any, Optional


COLOR_LABEL_MAP: dict[str, tuple[Optional[str], Optional[str]]] = {
    "green": ("green", "グリーン"),
    "yellow": ("yellow", "イエロー"),
    "blue": ("blue", "ブルー"),
    "red": ("red", "レッド"),
    "purple": ("purple", "パープル"),
    "none": (None, None),
    "": (None, None),
}


def safe_float(value: Any, default: float = 0.0) -> float:
    """空文字や不正値を安全に float 化する。"""
    try:
        if value in ("", None):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: Any) -> bool:
    """CSV由来の bool っぽい値を安全に bool 化する。"""
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value

    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False

    try:
        return bool(int(float(s)))
    except Exception:
        return False


def safe_int(value: Any, default: int = 0) -> int:
    """一般的な int 化。True/False も許容する。"""
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    s = str(value).strip()
    try:
        return int(float(s))
    except Exception:
        return default


def safe_int_flag(value: Any, default: int = 0) -> int:
    """
    CSV由来の 0/1, True/False, 'TRUE'/'False' を 0/1 に正規化する。
    int('False') のような事故を避けるための flag 専用変換。
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0

    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    if s in {"0", "false", "f", "no", "n"}:
        return 0

    try:
        return 1 if int(float(s)) != 0 else 0
    except Exception:
        return default


def get_str(row: dict[str, Any], key: str, default: str = "") -> str:
    """CSV row から文字列を安全に取得する。"""
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return str(value)


def normalize_lr_color_label(
    lr_color_label: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    lr_color_label ('Green' / 'green') から
    Lightroom 用の key/display を推定する。
    """
    if not lr_color_label:
        return None, None

    key = lr_color_label.strip().lower()
    return COLOR_LABEL_MAP.get(key, (key, None))


def normalize_lr_label_key(key: str) -> Optional[str]:
    """
    lr_labelcolor_key を Lightroom の許容 key に正規化する。
    許容外は None を返す。
    """
    if not key:
        return None

    normalized = key.strip().lower()
    return normalized if normalized in {"red", "yellow", "green", "blue", "purple"} else None


def compute_pick_from_csv(
    *,
    pick_mode: str,
    accepted_flag: int,
    secondary_flag: int,
    top_flag: int,
) -> int:
    """
    CSV の採用フラグ群から pick(0/1) を決める。

    pick_mode:
      - flags: accepted or secondary or top_flag
      - accepted: accepted only
      - accepted_or_secondary: accepted or secondary
      - none: always 0
    """
    mode = (pick_mode or "").strip().lower()

    if mode == "none":
        return 0
    if mode == "accepted":
        return 1 if accepted_flag == 1 else 0
    if mode == "accepted_or_secondary":
        return 1 if (accepted_flag == 1 or secondary_flag == 1) else 0

    # default: flags
    return 1 if (accepted_flag == 1 or secondary_flag == 1 or top_flag == 1) else 0
