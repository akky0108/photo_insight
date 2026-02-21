# src/evaluators/common/grade_contract.py
from __future__ import annotations

from typing import Optional

# =========================
# Discrete score contract
# =========================

# 許容する離散スコア
DISCRETE_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


# =========================
# Grade contract
# =========================

GRADE_BAD = "bad"
GRADE_POOR = "poor"
GRADE_FAIR = "fair"
GRADE_GOOD = "good"
GRADE_EXCELLENT = "excellent"

GRADE_ENUM = (
    GRADE_BAD,
    GRADE_POOR,
    GRADE_FAIR,
    GRADE_GOOD,
    GRADE_EXCELLENT,
)

# score → grade の唯一の対応表（SSOT）
SCORE_TO_GRADE = {
    0.0: GRADE_BAD,
    0.25: GRADE_POOR,
    0.5: GRADE_FAIR,
    0.75: GRADE_GOOD,
    1.0: GRADE_EXCELLENT,
}

# 過去互換・揺れ吸収（必要になったらここに追加していく）
# 例: 旧実装が "very_blurry"/"blurry" 等を返していた場合など
GRADE_NORMALIZE_MAP = {
    # blurriness旧表記の例（もし残っているなら吸収）
    "very_blurry": GRADE_BAD,
    "blurry": GRADE_POOR,
    "slightly_blurry": GRADE_FAIR,
    # ありがちな揺れ
    "excellent+": GRADE_EXCELLENT,
    "very_good": GRADE_GOOD,
    # 欠損系
    "none": None,
    "null": None,
    "": None,
}


def normalize_grade(grade: Optional[str]) -> Optional[str]:
    """
    grade を contract に正規化する。

    - grade が GRADE_ENUM のいずれかならそのまま
    - 既知の揺れ（GRADE_NORMALIZE_MAP）があれば吸収
    - 不明値は None に落とす（契約テストで検出可能）
    """
    if grade is None:
        return None

    g = str(grade).strip().lower()
    if not g:
        return None

    if g in GRADE_ENUM:
        return g

    if g in GRADE_NORMALIZE_MAP:
        return GRADE_NORMALIZE_MAP[g]

    return None


# =========================
# Eval status contract
# =========================

STATUS_OK = "ok"
STATUS_INVALID = "invalid"
STATUS_FALLBACK = "fallback"
STATUS_NOT_COMPUTED = "not_computed"

EVAL_STATUS_ENUM = {
    STATUS_OK,
    STATUS_INVALID,
    STATUS_FALLBACK,
    STATUS_NOT_COMPUTED,
}

# 過去互換・揺れ吸収用マップ
STATUS_NORMALIZE_MAP = {
    # old -> new
    "invalid_input": STATUS_INVALID,
    "invalid_data": STATUS_INVALID,
    "error": STATUS_INVALID,
    # fallback 系の揺れ吸収
    "fallback_used": STATUS_FALLBACK,
    "fallback_used_with_default": STATUS_FALLBACK,
    "default": STATUS_FALLBACK,
    "not_computed": STATUS_NOT_COMPUTED,
    "not_computed_with_default": STATUS_NOT_COMPUTED,  # ★事故ポイントを吸収
    "not_computed_default": STATUS_NOT_COMPUTED,  # 将来の揺れ予防（任意）
}


# =========================
# Utilities
# =========================


def normalize_score(score: Optional[float]) -> Optional[float]:
    """
    score を DISCRETE_SCORES に丸める（ズレ防止用）

    - None/変換不可: None
    - 既に離散値: そのまま
    - それ以外: 近い離散値へ丸める
    """
    if score is None:
        return None

    try:
        s = float(score)
    except Exception:
        return None

    if s in DISCRETE_SCORES:
        return s

    return min(DISCRETE_SCORES, key=lambda x: abs(x - s))


def score_to_grade(score: Optional[float]) -> Optional[str]:
    """
    score → grade 変換（唯一の正規ルート）
    """
    s = normalize_score(score)
    if s is None:
        return None
    return SCORE_TO_GRADE.get(s)


def normalize_eval_status(status: Optional[str]) -> str:
    """
    eval_status を contract に正規化

    方針：
    - 欠損/空: ok
    - 既知のenum: そのまま
    - 既知の揺れ: STATUS_NORMALIZE_MAP で吸収
    - 不明値: invalid に落とす
    """
    if not status:
        return STATUS_OK

    s = str(status).strip().lower()
    if not s:
        return STATUS_OK

    if s in EVAL_STATUS_ENUM:
        return s

    if s in STATUS_NORMALIZE_MAP:
        return STATUS_NORMALIZE_MAP[s]

    return STATUS_INVALID


def is_valid_grade(grade: Optional[str]) -> bool:
    return normalize_grade(grade) in GRADE_ENUM


def is_valid_score(score: Optional[float]) -> bool:
    return normalize_score(score) in DISCRETE_SCORES


def is_valid_eval_status(status: Optional[str]) -> bool:
    return normalize_eval_status(status) in EVAL_STATUS_ENUM
