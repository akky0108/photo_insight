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

    # fallback 系の揺れ吸収（★追加）
    "fallback_used": STATUS_FALLBACK,
    "fallback_used_with_default": STATUS_FALLBACK,

    "default": STATUS_FALLBACK,

    "not_computed": STATUS_NOT_COMPUTED,
    "not_computed_with_default": STATUS_NOT_COMPUTED,  # ★今回の事故ポイントを吸収
    "not_computed_default": STATUS_NOT_COMPUTED,       # 将来の揺れ予防（任意）
}


# =========================
# Utilities
# =========================

def normalize_score(score: Optional[float]) -> Optional[float]:
    """
    score を DISCRETE_SCORES に丸める（ズレ防止用）
    """
    if score is None:
        return None

    try:
        s = float(score)
    except Exception:
        return None

    if s in DISCRETE_SCORES:
        return s

    # 近い離散値に丸める
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
    """
    if not status:
        return STATUS_OK

    s = str(status).lower()

    if s in EVAL_STATUS_ENUM:
        return s

    if s in STATUS_NORMALIZE_MAP:
        return STATUS_NORMALIZE_MAP[s]

    # 不明値は invalid に落とす
    return STATUS_INVALID


def is_valid_grade(grade: Optional[str]) -> bool:
    return grade in GRADE_ENUM


def is_valid_score(score: Optional[float]) -> bool:
    return normalize_score(score) in DISCRETE_SCORES
