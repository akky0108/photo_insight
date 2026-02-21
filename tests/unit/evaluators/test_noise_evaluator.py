import numpy as np

from photo_insight.evaluators.noise_evaluator import NoiseEvaluator


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


def test_noise_evaluator_config_overrides_affect_discretize():
    cfg_a = {"noise": {"good_sigma": 0.010, "warn_sigma": 0.018}}
    cfg_b = {"noise": {"good_sigma": 0.020, "warn_sigma": 0.030}}

    ev_a = NoiseEvaluator(config=cfg_a)
    ev_b = NoiseEvaluator(config=cfg_b)

    sigma = 0.015
    score_a, grade_a = ev_a._score_discrete(sigma)
    score_b, grade_b = ev_b._score_discrete(sigma)

    # 閾値が緩い(b)ほどスコアは高く（または同等に）なりやすい
    assert score_b >= score_a
    assert grade_a in (
        "excellent",
        "good",
        "fair",
        "poor",
        "bad",
        "warn",
    )  # guard fallback含む
    assert grade_b in ("excellent", "good", "fair", "poor", "bad", "warn")


def test_noise_raw_is_negative_sigma_used_contract(monkeypatch):
    cfg = {"noise": {"good_sigma": 0.010, "warn_sigma": 0.018, "min_mask_ratio": 0.0}}
    ev = NoiseEvaluator(config=cfg)

    img = make_dummy_image(size=64, noise_level=5)
    fixed_sigma = 0.0123

    # 内部処理を固定化して evaluate の出力契約だけを見る
    monkeypatch.setattr(
        ev, "_to_luma01", lambda image: (np.zeros((64, 64), dtype=np.float32), "uint8")
    )
    monkeypatch.setattr(
        ev, "_downsample_long_edge", lambda luma01, le: (luma01, (64, 64))
    )
    monkeypatch.setattr(ev, "_residual", lambda luma01, sigma: np.zeros_like(luma01))
    monkeypatch.setattr(
        ev, "_build_mask", lambda luma01: np.ones_like(luma01, dtype=bool)
    )
    monkeypatch.setattr(ev, "_mad_sigma", lambda x: fixed_sigma)

    r = ev.evaluate(img)

    assert r["noise_sigma_used"] == fixed_sigma
    assert r["noise_raw"] == -fixed_sigma
