import numpy as np
import pytest
import cv2

from evaluators.color_balance_evaluator import ColorBalanceEvaluator


@pytest.fixture
def evaluator():
    return ColorBalanceEvaluator()


def test_color_balance_on_valid_image(evaluator):
    # 肌色に近い色合いの画像（ライトオレンジの単色）
    skin_like_image = np.full((256, 256, 3), (200, 160, 140), dtype=np.uint8)
    result = evaluator.evaluate(skin_like_image)

    assert isinstance(result, dict)
    assert "skin_tone_score" in result
    assert "white_balance_score" in result
    assert 0 <= result["skin_tone_score"] <= 1
    assert 0 <= result["white_balance_score"] <= 1


def test_color_balance_on_gray_image(evaluator):
    # 無彩色画像（白黒画像）
    gray_image = np.full((256, 256, 3), 128, dtype=np.uint8)
    result = evaluator.evaluate(gray_image)

    assert isinstance(result, dict)
    assert "skin_tone_score" in result
    assert "white_balance_score" in result
    assert 0 <= result["skin_tone_score"] <= 1
    assert 0 <= result["white_balance_score"] <= 1


def test_color_balance_on_invalid_input(evaluator):
    # 無効な入力（2D配列）
    with pytest.raises(ValueError):
        invalid_image = np.zeros((256, 256), dtype=np.uint8)
        evaluator.evaluate(invalid_image)
