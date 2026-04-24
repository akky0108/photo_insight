from photo_insight.pipelines.portrait_quality._internal.category import (
    PortraitCategory,
    classify_portrait_category,
)


def test_face_detected_is_portrait_face():
    assert (
        classify_portrait_category(
            face_detected=True,
            face_portrait_candidate=False,
        )
        == PortraitCategory.PORTRAIT_FACE
    )


def test_no_face_but_candidate_is_portrait_face():
    assert (
        classify_portrait_category(
            face_detected=False,
            face_portrait_candidate=True,
        )
        == PortraitCategory.PORTRAIT_FACE
    )


def test_full_body_is_portrait_body():
    assert (
        classify_portrait_category(
            face_detected=False,
            face_portrait_candidate=False,
            full_body_detected=True,
        )
        == PortraitCategory.PORTRAIT_BODY
    )


def test_non_face():
    assert (
        classify_portrait_category(
            face_detected=False,
            face_portrait_candidate=False,
            full_body_detected=False,
            shot_type="landscape",
        )
        == PortraitCategory.NON_FACE
    )
