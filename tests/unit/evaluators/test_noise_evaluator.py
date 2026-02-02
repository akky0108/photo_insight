import numpy as np
import pytest

from evaluators.noise_evaluator import NoiseEvaluator


def make_dummy_image(size=256, noise_level=5):
    """
    疑似ノイズ画像生成
    """
    base = np.ones((size, size), dtype=np.float32) * 0.5
    noise = np.random.normal(0, noise_level / 255.0, base.shape)
    img = np.clip(base + noise, 0, 1)

    return (img * 255).astype(np.uint8)


def test_noise_evaluator_basic_output():
    """
    通常評価で主要キーが揃うか
    """
    ev = NoiseEvaluator()
    img = make_dummy_image()

    r = ev.evaluate(img)

    # 必須キー
    assert "noise_score" in r
    assert "noise_raw" in r
    assert "noise_sigma_used" in r
    assert "noise_eval_status" in r

    # 値域
    assert 0.0 <= r["noise_score"] <= 1.0

    if r["noise_raw"] is not None:
        assert isinstance(r["noise_raw"], float)


def test_noise_evaluator_discrete_score():
    """
    score が5段階に収まるか
    """
    ev = NoiseEvaluator()
    img = make_dummy_image()

    r = ev.evaluate(img)

    allowed = {0.0, 0.25, 0.5, 0.75, 1.0}
    assert r["noise_score"] in allowed


def test_noise_evaluator_fallback():
    """
    異常入力時に fallback が動くか
    """
    ev = NoiseEvaluator()

    # 空画像
    img = np.zeros((0, 0), dtype=np.uint8)

    r = ev.evaluate(img)

    assert r["noise_eval_status"] == "fallback"
    assert "noise_raw" in r
    assert r["noise_score"] == ev.fallback_score
