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
    ※ 今回は「技術で点が伸びすぎない」前提になったので、星はやや“厳しめ寄り”のまま据え置き。
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


def format_score(x: Any) -> float:
    """
    表示用に丸める（accepted_reason / lr_keywords で統一表記に使う）
    """
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return 0.0


def _infer_secondary_from_reason(reason: str) -> int:
    if not reason:
        return 0
    s = str(reason).strip()
    return (
        1
        if (
            s.startswith("SEC:")
            or s.startswith("SEC-RESCUE:")
            or s.startswith("ACC-SEC-FILL:")
        )
        else 0
    )


def _infer_green_from_reason(reason: str) -> int:
    """
    accepted_flag が欠けていても reason から Green 相当を推定する保険。
    """
    if not reason:
        return 0
    s = str(reason).strip()
    # build_reason_pro は "portrait group=... rank=... o=..." なので categoryで始まる
    if s.startswith("portrait ") or s.startswith("non_face "):
        return 1
    # 旧形式互換
    if "rank=" in s and "thr=" in s and ("overall=" in s or "o=" in s):
        return 1
    if s.startswith("ACC-FILL:") or s.startswith("ACC:") or s.startswith("ACC-"):
        return 1
    return 0


def choose_color_label(
    *,
    accepted_flag: int,
    secondary_flag: int,
    rating: int,
    top_flag: int,
    face_in_focus: bool,
    eye_state: str,
) -> str:
    """
    Lightroom color label の運用ルール（今回の方針に最適化）
    Green  : 本採用（accepted_flag=1）
    Yellow : セカンダリ採用（secondary_accept_flag=1） or 顔ピント良好で要検討
    Blue   : 上位候補（flag=1）
    Red    : 明確な問題（半目/閉眼/★1以下 & 顔ピントも悪い）
    None   : その他
    """
    # 目状態は最優先（撮影の判断として強い）
    # - half: 原則NG → Red
    # - closed: 注意 → Yellow（ただしGreenが付くならGreen優先）
    if accepted_flag == 1:
        return "Green"

    if eye_state == "half":
        return "Red"

    if secondary_flag == 1:
        return "Yellow"

    if eye_state == "closed":
        return "Yellow"

    # 明確な失敗
    if rating <= 1 and not face_in_focus:
        return "Red"

    # まず候補を青で拾う（Blue=“見る”）
    if top_flag == 1:
        return "Blue"

    # 顔ピント良好なら要検討として黄
    if face_in_focus:
        return "Yellow"

    return ""


def shorten_reason_for_lr(reason: str, *, max_len: int = 90) -> str:
    """
    Lightroom のキーワード用に短文化する。
    長い場合は「先頭 + … + 末尾」を残して、tail も見えるようにする。
    """
    if not reason:
        return ""

    s = str(reason)

    # 邪魔な装飾の圧縮
    s = s.replace(" | ", " / ")
    s = s.replace("overall=", "o=")
    s = s.replace("score_face=", "f=")
    s = s.replace("score_composition=", "c=")
    s = s.replace("score_technical=", "t=")

    # build_reason_pro 由来の "tags=" を見やすく（tags=a,b,c → tags=a/b/c）
    if "tags=" in s:
        head, tail = s.split("tags=", 1)
        tail = tail.replace(",", "/").replace(" ", "")
        s = head + "tags=" + tail

    # 連続空白の除去
    s = " ".join(s.split())

    if max_len <= 0:
        return ""

    # 短ければそのまま
    if len(s) <= max_len:
        return s

    # max_len が小さすぎる場合は素直に先頭切り
    if max_len <= 10:
        return s[:max_len]

    # --- ここが tail を残す本体 ---
    # 末尾を確実に残す（最低20文字 or 全体の1/4）
    tail_len = max(20, max_len // 4)
    # ただし max_len を超えないように
    tail_len = min(tail_len, max_len - 2)  # "…" + head の余地
    head_len = max_len - 1 - tail_len  # "…" 1文字分

    return s[:head_len] + "…" + s[-tail_len:]


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

    # ★矛盾吸収：reason から推定
    inferred_green = _infer_green_from_reason(reason_norm) if accepted_flag == 0 else 0
    inferred_secondary = (
        _infer_secondary_from_reason(reason_norm)
        if (accepted_flag == 0 and secondary_flag == 0)
        else 0
    )

    effective_accepted_flag = 1 if accepted_flag == 1 or inferred_green == 1 else 0
    effective_secondary_flag = (
        1 if secondary_flag == 1 or inferred_secondary == 1 else 0
    )

    # eye_state は acceptance 側で付く（無ければ空）
    eye_state = str(row.get("eye_state") or "").strip().lower()

    face_detected = safe_bool(row.get("face_detected"))
    face_sharp_score = safe_float(row.get("face_sharpness_score"))
    face_sharp_100 = to_0_100(face_sharp_score)
    face_in_focus = bool(
        face_detected and face_sharp_100 >= FACE_SHARPNESS_FOCUS_THRESHOLD
    )

    label = choose_color_label(
        accepted_flag=effective_accepted_flag,
        secondary_flag=effective_secondary_flag,
        rating=rating,
        top_flag=top_flag,
        face_in_focus=face_in_focus,
        eye_state=eye_state,
    )

    row["lr_rating"] = rating
    row["lr_color_label"] = label
    row["lr_labelcolor_key"] = to_labelcolor_key(label)
    row["lr_label_display"] = to_label_display_from_key(row["lr_labelcolor_key"])

    # keywords:
    # - Green/Yellow は reason を入れる
    # - Blue は軽いヒントだけ（長文は邪魔）
    if effective_accepted_flag == 1 or effective_secondary_flag == 1:
        row["lr_keywords"] = shorten_reason_for_lr(reason_norm, max_len=keyword_max_len)
    elif label == "Blue":
        # Blue は「見る理由」を短く
        # 例: "CAND o=78.1"
        row["lr_keywords"] = f"CAND o={format_score(overall)}"
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
