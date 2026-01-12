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


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def score_to_rating(overall: float) -> int:
    """
    overall_score (0–100) → Lightroom 星評価 (0–5)

    運用意図:
    ★★★★★ (5): そのまま納品・公開・代表作候補（確実に残す）
    ★★★★☆ (4): 非常に良い。用途次第で採用
    ★★★☆☆ (3): 記録・素材として有用（削らない）
    ★★☆☆☆ (2): 微妙だが状況証拠として残す可能性あり
    ★☆☆☆☆ (1): 基本不要だが誤検出・検証用
    ☆☆☆☆☆ (0): 自動削除候補

    ※ 閾値は「日ごとの分布」よりも
       人間の最終判断基準として安定させる目的で固定値。
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


def choose_color_label(category: str, accepted_flag: int, secondary_flag: int, rating: int) -> str:
    """
    Lightroom color label の運用ルール

    Green  : 本採用（accepted_flag=1）
    Yellow : セカンダリ採用（secondary_accept_flag=1）
    Red    : 問題あり（★1以下）
    None   : その他
    """
    # 本採用が最優先
    if accepted_flag == 1:
        return "Green"

    # セカンダリ採用
    if secondary_flag == 1:
        return "Yellow"

    # 明らかに微妙（低レーティング）
    if rating <= 1:
        return "Red"

    # それ以外は色なし（星だけで判断）
    return ""


def shorten_reason_for_lr(reason: str, max_len: int = 90, prefix: str = "ACC:") -> str:
    """
    Lightroom のキーワード用に短文化する。
    - prefix で ACC: / SEC: などを付けられる
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

    s = " ".join(s.split())

    if prefix:
        s = prefix + s

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
    """
    row に Lightroom 用フィールドを in-place で付与する。

    必要入力（row内）:
      - overall_score
      - category（portrait/non_face）
      - accepted_flag
      - accepted_reason（accepted=1 のときだけ有効）

    付与出力:
      - lr_rating
      - lr_color_label
      - lr_labelcolor_key
      - lr_label_display
      - lr_keywords
    """
    overall = safe_float(row.get("overall_score"))
    rating = score_to_rating(overall)

    accepted_flag = int(row.get("accepted_flag", 0) or 0)
    secondary_flag = int(row.get("secondary_accept_flag", 0) or 0)  # ★追加
    category = str(row.get("category", "") or "")

    label = choose_color_label(category, accepted_flag, secondary_flag, rating)

    row["lr_rating"] = rating
    row["lr_color_label"] = label  # 既存互換として残す（任意）
    row["lr_labelcolor_key"] = to_labelcolor_key(label)
    row["lr_label_display"] = to_label_display_from_key(row["lr_labelcolor_key"])

    reason_primary = row.get("accepted_reason", "") or ""
    reason_secondary = row.get("secondary_accept_reason", "") or ""

    if accepted_flag == 1:
        # 本採用 → ACC:
        row["lr_keywords"] = shorten_reason_for_lr(
            reason_primary, max_len=keyword_max_len, prefix="ACC:"
        )
    elif secondary_flag == 1:
        # セカンダリ → SEC:
        # secondary_accept_reason 側にすでに "SEC:" が含まれている場合は prefix="" でもOK
        # ここでは prefix="SEC:" を付けて分かりやすくしている
        row["lr_keywords"] = shorten_reason_for_lr(
            reason_secondary, max_len=keyword_max_len, prefix="SEC:"
        )
    else:
        row["lr_keywords"] = ""


# ここを追加
def apply_lightroom_fields_to_rows(rows: List[Dict[str, Any]], *, keyword_max_len: int = 90) -> None:
    """
    rows 全体に apply_lightroom_fields を適用するヘルパー。
    バッチ処理側はこれを1発呼べばよい。
    """
    for r in rows:
        apply_lightroom_fields(r, keyword_max_len=keyword_max_len)
