# src/batch_processor/evaluation_rank/lightroom.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List


# Lightroom（日本語ラベルセット想定）: 表示名マップ
LR_LABEL_DISPLAY_JA = {
    "red": "レッド",
    "yellow": "イエロー",
    "green": "グリーン",
    "blue": "ブルー",
    "purple": "パープル",
    "none": "",
    "": "",
}

# 顔ピント「良し」とみなすしきい値（0〜100スケール）
FACE_SHARPNESS_FOCUS_THRESHOLD = 70.0


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def safe_bool(value: Any) -> bool:
    """
    CSVの TRUE/FALSE/1/0/yes/no などをざっくり bool 化。
    """
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    # 最後に数値っぽいものだけ拾う
    try:
        return bool(int(float(s)))
    except Exception:
        return False


def safe_int_flag(value: Any) -> int:
    """
    CSV由来の 0/1, True/False, "TRUE"/"False" を 0/1 に正規化する。
    int("False") 事故を確実に回避するため、ここ以外で int(...) しない。
    """
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0

    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n"):
        return 0

    # それでもダメなら最後に数値変換を試す（例: "2" / "1.0"）
    try:
        return 1 if int(float(s)) != 0 else 0
    except Exception:
        return 0


def to_0_100(v: Any) -> float:
    """
    0〜1 でも 0〜100 でも入ってきてもよいように正規化。
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if v < 0:
        return 0.0
    if v <= 1.0:
        return v * 100.0
    return v


def score_to_rating(overall: float) -> int:
    """
    overall_score (0–100) → Lightroom 星評価 (0–5)
    """
    if overall >= 85:
        return 5
    if overall >= 75:
        return 4
    if overall >= 65:
        return 3
    if overall >= 55:
        return 2
    if overall >= 45:
        return 1
    return 0


def choose_color_label(
    *,
    accepted_flag: int,
    secondary_flag: int,
    rating: int,
    top_flag: int,
    face_in_focus: bool,
) -> str:
    """
    Lightroom color label の運用ルール
    Green  : 本採用（accepted_flag=1）
    Yellow : セカンダリ採用（secondary_accept_flag=1） or 顔ピント良好で要検討
    Blue   : 上位だがまだ保留（flag=1 & rating>=2）
    Red    : 問題あり（★1以下 & 顔ピントも悪い）
    None   : その他
    """
    if accepted_flag == 1:
        return "Green"
    if secondary_flag == 1:
        return "Yellow"
    if rating <= 1 and not face_in_focus:
        return "Red"
    if top_flag == 1 and rating >= 2:
        return "Blue"
    if face_in_focus:
        return "Yellow"
    return ""


def shorten_reason_for_lr(reason: str, *, max_len: int = 90) -> str:
    """
    Lightroom のキーワード用に短文化する（prefixは reason 側に入っている前提）。
    accepted_reason が ACC:/SEC:/SEC-RESCUE: を含むので、ここで付け足さない。
    """
    if not reason:
        return ""

    s = str(reason)

    # 表現を圧縮（LRで邪魔にならない）
    s = s.replace(" | ", " / ")
    s = s.replace("(tech=", "t=")
    s = s.replace(" face=", " f=")
    s = s.replace(" comp=", " c=")
    s = s.replace("overall=", "o=")

    # 連続空白の除去
    s = " ".join(s.split())

    return s[:max_len]


def to_labelcolor_key(color: str) -> str:
    """
    CSV内の lr_color_label ("Green" etc) を XMPの photoshop:LabelColor 用キーに寄せる。
    """
    if not color:
        return "none"
    c = str(color).strip().lower()
    if c in {"red", "yellow", "green", "blue", "purple"}:
        return c
    return "none"


def to_label_display_from_key(label_key: str) -> str:
    return LR_LABEL_DISPLAY_JA.get(label_key, "")


def apply_lightroom_fields(row: Dict[str, Any], *, keyword_max_len: int = 90) -> None:
    overall = safe_float(row.get("overall_score"))
    rating = score_to_rating(overall)

    accepted_flag = safe_int_flag(row.get("accepted_flag", 0))
    secondary_flag = safe_int_flag(row.get("secondary_accept_flag", 0))
    top_flag = safe_int_flag(row.get("flag", 0))

    # ★理由は accepted_reason 1本（A統一）
    reason = row.get("accepted_reason", "") or ""
    reason_norm = str(reason).strip()

    # ★最後の砦：reason から secondary を推定して矛盾を吸収
    inferred_secondary = 1 if (
        accepted_flag == 0
        and secondary_flag == 0
        and (reason_norm.startswith("SEC:") or reason_norm.startswith("SEC-RESCUE:"))
    ) else 0
    effective_secondary_flag = 1 if secondary_flag == 1 or inferred_secondary == 1 else 0

    face_detected = safe_bool(row.get("face_detected"))
    face_sharp_score = safe_float(row.get("face_sharpness_score"))
    face_sharp_100 = to_0_100(face_sharp_score)
    face_in_focus = bool(face_detected and face_sharp_100 >= FACE_SHARPNESS_FOCUS_THRESHOLD)

    label = choose_color_label(
        accepted_flag=accepted_flag,
        secondary_flag=effective_secondary_flag,  # ★ここ
        rating=rating,
        top_flag=top_flag,
        face_in_focus=face_in_focus,
    )

    row["lr_rating"] = rating
    row["lr_color_label"] = label
    row["lr_labelcolor_key"] = to_labelcolor_key(label)
    row["lr_label_display"] = to_label_display_from_key(row["lr_labelcolor_key"])

    # ★keywords：accepted または（推定込み）secondary のときだけ reason を載せる
    if accepted_flag == 1 or effective_secondary_flag == 1:
        row["lr_keywords"] = shorten_reason_for_lr(reason_norm, max_len=keyword_max_len)
    else:
        row["lr_keywords"] = ""


def apply_lightroom_fields_to_rows(
    rows: List[Dict[str, Any]],
    *,
    keyword_max_len: int = 90,
) -> None:
    """
    rows 全体に apply_lightroom_fields を適用するヘルパー。
    """
    for r in rows:
        apply_lightroom_fields(r, keyword_max_len=keyword_max_len)
