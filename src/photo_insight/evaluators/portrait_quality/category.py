# src/photo_insight/pipelines/portrait_quality/_internal/category.py

from enum import Enum


class PortraitCategory(str, Enum):
    PORTRAIT_FACE = "portrait_face"
    PORTRAIT_BODY = "portrait_body"
    NON_FACE = "non_face"


def classify_portrait_category(
    *,
    face_detected: bool,
    face_portrait_candidate: bool,
    full_body_detected: bool | None = None,
    shot_type: str | None = None,
) -> PortraitCategory:
    """
    Portrait分類（責務: 写真タイプの決定のみ）

    優先順位:
    1. 顔主体 → portrait_face
    2. 全身/人物構図 → portrait_body
    3. その他 → non_face

    NOTE:
    - face_detected=False でも face_portrait_candidate=True なら face 扱い
    - face_detected=False を non_face に直結しない
    """

    # normalize
    shot_type_norm = (shot_type or "").lower()

    # 1. 顔ポートレート
    if face_detected or face_portrait_candidate:
        return PortraitCategory.PORTRAIT_FACE

    # 2. 全身ポートレート
    if full_body_detected:
        return PortraitCategory.PORTRAIT_BODY

    if shot_type_norm in {"full_body", "seated"}:
        return PortraitCategory.PORTRAIT_BODY

    # 3. 非ポートレート
    return PortraitCategory.NON_FACE


def to_legacy_category(category: PortraitCategory) -> str:
    """
    evaluation_rank 互換用
    """
    if category in {
        PortraitCategory.PORTRAIT_FACE,
        PortraitCategory.PORTRAIT_BODY,
    }:
        return "portrait"

    return "non_face"
