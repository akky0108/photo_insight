import numpy as np
import pytest
from unittest.mock import MagicMock
from photo_insight.evaluators.fullbody_composition_evaluator import (
    FullBodyCompositionEvaluator,
)

# === Fixtures ===


@pytest.fixture
def evaluator():
    return FullBodyCompositionEvaluator()


@pytest.fixture
def dummy_image():
    # 640x480 黒画像
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def valid_keypoints():
    # 17点、画面中央あたりにある
    return [[320, 240] for _ in range(17)]


@pytest.fixture
def invalid_keypoints():
    # 17個、全部None
    return [None for _ in range(17)]


# === 正常系テスト ===


def test_evaluate_body_position_valid(evaluator, dummy_image, valid_keypoints):
    score = evaluator.evaluate_body_position(dummy_image, valid_keypoints)
    assert score == 1.0


def test_evaluate_body_balance_valid(evaluator, dummy_image, valid_keypoints):
    score = evaluator.evaluate_body_balance(dummy_image, valid_keypoints)
    assert 0.0 <= score <= 1.0


def test_evaluate_pose_dynamics_valid(evaluator, dummy_image, valid_keypoints):
    score = evaluator.evaluate_pose_dynamics(dummy_image, valid_keypoints)
    assert 0.0 <= score <= 1.0


def test_classify_group(evaluator):
    assert evaluator.classify_group(0.9) == "high_quality"
    assert evaluator.classify_group(0.7) == "medium_quality"
    assert evaluator.classify_group(0.4) == "low_quality"


def test_calculate_final_score(evaluator):
    sample_scores = {
        evaluator.BODY_POSITION: 0.8,
        evaluator.BODY_BALANCE: 0.6,
        evaluator.POSE_DYNAMICS: 0.7,
    }
    score = evaluator.calculate_final_score(sample_scores)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# === 異常系テスト ===


def test_evaluate_body_position_invalid(evaluator, dummy_image, invalid_keypoints):
    score = evaluator.evaluate_body_position(dummy_image, invalid_keypoints)
    assert score == 0.0


def test_evaluate_body_balance_with_insufficient_keypoints(evaluator, dummy_image):
    insufficient_keypoints = [[320, 240] for _ in range(5)]  # 5個だけ
    score = evaluator.evaluate_body_balance(dummy_image, insufficient_keypoints)
    assert score == 0.0


def test_evaluate_pose_dynamics_with_no_valid_pairs(evaluator, dummy_image):
    broken_keypoints = [None for _ in range(17)]
    score = evaluator.evaluate_pose_dynamics(dummy_image, broken_keypoints)
    assert score == 0.0


# === 統合テスト（evaluate全部） ===


def test_evaluate_full_flow(evaluator, dummy_image, valid_keypoints):
    results = evaluator.evaluate(dummy_image, valid_keypoints)

    assert "composition_fullbody_score" in results
    assert "body_position_score" in results
    assert "body_balance_score" in results
    assert "pose_dynamics_score" in results
    assert "group_id" in results
    assert "subgroup_id" in results

    assert 0.0 <= results["composition_fullbody_score"] <= 1.0
    assert results["group_id"] in ["high_quality", "medium_quality", "low_quality"]


# === ログモックテスト ===


def test_logger_warning_when_invalid_keypoints(dummy_image):
    logger_mock = MagicMock()
    evaluator = FullBodyCompositionEvaluator(logger=logger_mock)

    invalid_keypoints = [None for _ in range(17)]
    evaluator.evaluate_body_position(dummy_image, invalid_keypoints)

    logger_mock.warning.assert_called_once()
