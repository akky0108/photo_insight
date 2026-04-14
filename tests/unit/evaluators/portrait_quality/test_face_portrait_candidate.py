from __future__ import annotations

try:
    from photo_insight.evaluators.portrait_quality.portrait_quality_evaluator import (
        PortraitQualityEvaluator,
    )
except ImportError:
    from evaluators.portrait_quality.portrait_quality_evaluator import (  # type: ignore
        PortraitQualityEvaluator,
    )


def _make_evaluator() -> PortraitQualityEvaluator:
    evaluator = PortraitQualityEvaluator.__new__(PortraitQualityEvaluator)
    return evaluator


def test_detect_face_portrait_candidate_true_when_face_detected() -> None:
    evaluator = _make_evaluator()

    actual = evaluator._detect_face_portrait_candidate(
        face_detected=True,
        face_boxes=[{"box": [0, 0, 10, 10]}],
        results={},
    )

    assert actual is True


def test_detect_face_portrait_candidate_true_for_close_up_without_full_body() -> None:
    evaluator = _make_evaluator()

    actual = evaluator._detect_face_portrait_candidate(
        face_detected=False,
        face_boxes=[],
        results={
            "shot_type": "close_up",
            "full_body_detected": False,
        },
    )

    assert actual is True


def test_detect_face_portrait_candidate_true_for_upper_body_without_full_body() -> None:
    evaluator = _make_evaluator()

    actual = evaluator._detect_face_portrait_candidate(
        face_detected=False,
        face_boxes=[],
        results={
            "shot_type": "upper_body",
            "full_body_detected": False,
        },
    )

    assert actual is True


def test_detect_face_portrait_candidate_false_when_no_face_and_no_hint() -> None:
    evaluator = _make_evaluator()

    actual = evaluator._detect_face_portrait_candidate(
        face_detected=False,
        face_boxes=[],
        results={
            "shot_type": "",
            "full_body_detected": False,
        },
    )

    assert actual is False


def test_detect_face_portrait_candidate_false_for_full_body() -> None:
    evaluator = _make_evaluator()

    actual = evaluator._detect_face_portrait_candidate(
        face_detected=False,
        face_boxes=[],
        results={
            "shot_type": "upper_body",
            "full_body_detected": True,
        },
    )

    assert actual is False
