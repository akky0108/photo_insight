# tests/unit/evaluators/test_local_contrast_evaluator.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

# プロジェクトの import 事情（src/ を PYTHONPATH に乗せるなど）により変わるので、
# まずは通常 import を試し、ダメなら pytest.ini / conftest.py 側で調整してください。
from evaluators.local_contrast_evaluator import LocalContrastEvaluator


def _make_texture_uint8(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """
    それなりにコントラスト/テクスチャがある合成画像（uint8 0..255）
    単なる乱数より「ブロック std」が安定しやすいように、緩い勾配＋ノイズを混ぜる。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    grad = (xx / max(w - 1, 1)) * 160.0 + (yy / max(h - 1, 1)) * 40.0  # 緩い勾配
    noise = rng.normal(loc=0.0, scale=12.0, size=(h, w))               # 細かい揺らぎ
    img = grad + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _to_float01(img_u8: np.ndarray) -> np.ndarray:
    """uint8 0..255 -> float32 0..1"""
    return (img_u8.astype(np.float32) / 255.0).astype(np.float32)


@pytest.mark.unit
def test_local_contrast_scale_invariance_score_uint8_vs_float01():
    """
    ★最重要テスト
    入力スケールが 0..255 と 0..1 で異なっても、
    score（5段階）は同等になること。
    raw はスケール依存なので一致しなくてOK。
    """
    evaluator = LocalContrastEvaluator(
        block_size=32,
        min_blocks=9,
        ignore_low_dynamic_blocks=True,
        # スケール非依存の比率閾値（実運用に合わせて）
        raw_floor=0.010,
        raw_ceil=0.060,
        gamma=0.9,
        low_dynamic_threshold=0.02,
        robust_p_low=5.0,
        robust_p_high=95.0,
    )

    img_u8 = _make_texture_uint8(256, 256, seed=42)
    img_f01 = _to_float01(img_u8)

    r_u8 = evaluator.evaluate(img_u8)
    r_f01 = evaluator.evaluate(img_f01)

    assert r_u8["local_contrast_eval_status"] in ("ok", "fallback_used")
    assert r_f01["local_contrast_eval_status"] in ("ok", "fallback_used")

    # スケール非依存のはずなので score は一致（少なくとも同じ離散値）してほしい
    assert r_u8["local_contrast_score"] == r_f01["local_contrast_score"], (
        f"score mismatch: uint8={r_u8['local_contrast_score']} float01={r_f01['local_contrast_score']} "
        f"raw uint8={r_u8['local_contrast_raw']}, raw float01={r_f01['local_contrast_raw']}"
    )

    # score は規定の5段階のみ
    assert r_u8["local_contrast_score"] in (0.0, 0.25, 0.5, 0.75, 1.0)
    assert r_f01["local_contrast_score"] in (0.0, 0.25, 0.5, 0.75, 1.0)


@pytest.mark.unit
def test_local_contrast_returns_ok_on_textured_image():
    """
    テクスチャがある画像なら ok で返ることが多い（環境差でfallbackになる場合もあるので緩めに）。
    """
    evaluator = LocalContrastEvaluator(block_size=32, min_blocks=9)
    img = _make_texture_uint8(256, 256, seed=1)
    r = evaluator.evaluate(img)

    assert "local_contrast_raw" in r
    assert "local_contrast_score" in r
    assert "local_contrast_std" in r
    assert "local_contrast_eval_status" in r
    assert "local_contrast_fallback_reason" in r
    assert "success" in r

    assert r["local_contrast_eval_status"] in ("ok", "fallback_used")
    assert r["local_contrast_score"] in (0.0, 0.25, 0.5, 0.75, 1.0)


@pytest.mark.unit
def test_local_contrast_fallback_on_too_small_image():
    """
    画像が block_size より小さい場合は fallback_used になること。
    """
    evaluator = LocalContrastEvaluator(block_size=32, min_blocks=9)

    img_small = _make_texture_uint8(16, 16, seed=0)  # block_size 未満
    r = evaluator.evaluate(img_small)

    assert r["local_contrast_eval_status"] == "fallback_used"
    assert "image_too_small_for_blocks" in r["local_contrast_fallback_reason"]
    assert r["success"] is True


@pytest.mark.unit
def test_local_contrast_handles_near_constant_image_with_reason():
    """
    ほぼ単色（低ダイナミック）の場合でも例外にならず、fallback または bad で理由が残ること。
    """
    evaluator = LocalContrastEvaluator(
        block_size=32,
        min_blocks=9,
        ignore_low_dynamic_blocks=True,
        low_dynamic_threshold=0.02,
    )

    img_const_u8 = np.full((256, 256), 128, dtype=np.uint8)
    r = evaluator.evaluate(img_const_u8)

    # “ok” で返ることは通常少ない（ブロック除外で insufficient_blocks になりがち）
    assert r["local_contrast_eval_status"] in ("fallback_used", "invalid_input", "ok")
    if r["local_contrast_eval_status"] == "fallback_used":
        assert r["local_contrast_fallback_reason"] != ""
        assert r["success"] is True


@pytest.mark.unit
def test_local_contrast_invalid_input_type():
    evaluator = LocalContrastEvaluator()
    r = evaluator.evaluate("not ndarray")  # type: ignore[arg-type]
    assert r["local_contrast_eval_status"] == "invalid_input"
    assert r["success"] is False
    assert "type_not_ndarray" in r["local_contrast_fallback_reason"]


def test_scale_invariant_float_vs_uint8():

    from evaluators.local_contrast_evaluator import LocalContrastEvaluator

    ev = LocalContrastEvaluator()

    # テスト用パターン
    img_uint8 = np.random.randint(
        0, 256, size=(256, 256, 3), dtype=np.uint8
    )

    img_float = img_uint8.astype(np.float32) / 255.0

    r1 = ev.evaluate(img_uint8)
    r2 = ev.evaluate(img_float)

    assert r1["local_contrast_score"] == r2["local_contrast_score"]
