import pytest
import numpy as np

# -------------------------
# Robust imports (repo layout drift friendly)
# -------------------------
try:
    # preferred (if src/ is on PYTHONPATH)
    from src.evaluators.contrast_evaluator import ContrastEvaluator  # type: ignore
except Exception:
    from photo_insight.evaluators.contrast_evaluator import ContrastEvaluator  # type: ignore  # noqa: E501

try:
    from src.evaluators.common.grade_contract import (  # type: ignore
        normalize_eval_status,
        GRADE_ENUM,
        EVAL_STATUS_ENUM,
        STATUS_INVALID,
        STATUS_OK,
        score_to_grade,
    )
except Exception:
    from photo_insight.evaluators.common.grade_contract import (  # type: ignore
        normalize_eval_status,
        GRADE_ENUM,
        EVAL_STATUS_ENUM,
        STATUS_INVALID,
        STATUS_OK,
        score_to_grade,
    )


def _make_std30_gray_u8() -> np.ndarray:
    """
    0 と 60 を半分ずつ: mean=30, diff=±30 -> std=30
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, :50] = 0
    img[:, 50:] = 60
    return img


def _inject_face_cfg(eval_cfg: dict) -> dict:
    """
    PortraitQualityEvaluator の _subcfg("face_contrast") 相当:
      {"contrast": cfg["face_contrast"]}
    """
    spec = (eval_cfg or {}).get("face_contrast")
    return {"contrast": spec} if isinstance(spec, dict) else {}


def _assert_contract_for_contrast_result(r: dict) -> None:
    """
    ④: CSV契約テスト（最小版）
      - grade ∈ GRADE_ENUM ∪ {None}
      - eval_status ∈ EVAL_STATUS_ENUM
    """
    assert "contrast_eval_status" in r
    assert normalize_eval_status(r.get("contrast_eval_status")) in EVAL_STATUS_ENUM

    # grade は None 許容（評価失敗・未計算等）
    assert "contrast_grade" in r
    g = r.get("contrast_grade")
    assert (g is None) or (g in GRADE_ENUM)

    # score があるなら grade と整合する（score_to_grade がSSOT）
    if "contrast_score" in r and r.get("contrast_score") is not None:
        assert r.get("contrast_grade") in (
            None,
            score_to_grade(r.get("contrast_score")),
        )


@pytest.mark.parametrize(
    "cfg, expected_score, expected_grade",
    [
        # A: 30 は fair(0.5)
        (
            {
                "contrast": {
                    "discretize_thresholds_raw": {
                        "poor": 10.0,
                        "fair": 20.0,
                        "good": 40.0,
                        "excellent": 80.0,
                    }
                }
            },
            0.5,
            "fair",
        ),
        # B: 30 は good(0.75)
        (
            {
                "contrast": {
                    "discretize_thresholds_raw": {
                        "poor": 5.0,
                        "fair": 10.0,
                        "good": 20.0,
                        "excellent": 100.0,
                    }
                }
            },
            0.75,
            "good",
        ),
    ],
)
def test_contrast_evaluator_config_thresholds_affect_score(
    cfg, expected_score, expected_grade
):
    """
    config の discretize_thresholds_raw により score/grade が変わることを保証する。
    std が約30になる画像を作って判定を安定化する。
    """
    img = _make_std30_gray_u8()
    ev = ContrastEvaluator(config=cfg)
    r = ev.evaluate(img)

    assert normalize_eval_status(r.get("contrast_eval_status")) == STATUS_OK
    assert r["contrast_score"] == expected_score
    assert r["contrast_grade"] == expected_grade
    assert r["contrast_raw"] is not None

    _assert_contract_for_contrast_result(r)


def test_contrast_float01_input_scaled_to_gray255():
    """
    float01 入力でも ensure_gray255 により 0..255 に吸収され、
    raw(stddev) が uint8 と同等になることを保証する。
    """
    img = np.zeros((100, 100), dtype=np.float32)
    img[:, :50] = 0.0
    img[:, 50:] = 60.0 / 255.0  # float01

    ev = ContrastEvaluator()
    r = ev.evaluate(img)

    # std(0,60)=30 のはず（許容つき）
    assert 29.0 < float(r["contrast_raw"]) < 31.0
    assert normalize_eval_status(r.get("contrast_eval_status")) == STATUS_OK

    _assert_contract_for_contrast_result(r)


def test_contrast_thresholds_sorted():
    """
    load_thresholds_sorted の単調性保証（並び替え）が効いていることをテストする。
    """
    cfg = {
        "contrast": {
            "discretize_thresholds_raw": {
                "poor": 50,
                "fair": 10,
                "good": 40,
                "excellent": 20,
            }
        }
    }
    ev = ContrastEvaluator(config=cfg)

    assert ev.t_poor <= ev.t_fair <= ev.t_good <= ev.t_excellent


def test_face_contrast_thresholds_are_separated_by_injection():
    """
    face_contrast の閾値分離は evaluator の metric_key 切替ではなく、
    外側での config 注入（{"contrast": cfg["face_contrast"]}）で実現する設計。
    その注入で score/grade が変わることを保証する。
    """
    img = _make_std30_gray_u8()

    cfg = {
        "contrast": {  # global: 30 => fair(0.5)
            "discretize_thresholds_raw": {
                "poor": 10.0,
                "fair": 20.0,
                "good": 40.0,
                "excellent": 80.0,
            }
        },
        "face_contrast": {  # face: 30 => good(0.75)
            "discretize_thresholds_raw": {
                "poor": 5.0,
                "fair": 10.0,
                "good": 20.0,
                "excellent": 100.0,
            }
        },
    }

    # global
    ev_global = ContrastEvaluator(config=cfg, metric_key="contrast")
    r_g = ev_global.evaluate(img)
    assert normalize_eval_status(r_g.get("contrast_eval_status")) == STATUS_OK
    assert r_g["contrast_score"] == 0.5
    assert r_g["contrast_grade"] == "fair"
    assert r_g.get("contrast_thresholds_metric_key") == "contrast"
    _assert_contract_for_contrast_result(r_g)

    # face (injection)
    face_cfg = _inject_face_cfg(cfg)
    ev_face = ContrastEvaluator(config=face_cfg, metric_key="contrast")
    r_f = ev_face.evaluate(img)

    # 出力キーは contrast_* のまま（prefix付与は mapper の責務）
    assert normalize_eval_status(r_f.get("contrast_eval_status")) == STATUS_OK
    assert r_f["contrast_score"] == 0.75
    assert r_f["contrast_grade"] == "good"
    assert r_f.get("contrast_thresholds_metric_key") == "contrast"
    _assert_contract_for_contrast_result(r_f)


def test_contrast_empty_returns_invalid():
    ev = ContrastEvaluator()
    img = np.zeros((0, 0), dtype=np.uint8)

    r = ev.evaluate(img)

    # 実装揺れ（invalid_input等）があっても contract SSOT で正規化して見る
    assert normalize_eval_status(r.get("contrast_eval_status")) == STATUS_INVALID
    assert r.get("success") in (False, 0, None)  # 実装差の吸収
    assert r["contrast_score"] == 0.0
    assert r["contrast_grade"] == "bad"

    _assert_contract_for_contrast_result(r)


def test_contrast_includes_thresholds_payload():
    img = _make_std30_gray_u8()
    ev = ContrastEvaluator()
    r = ev.evaluate(img)

    assert "contrast_thresholds_raw" in r
    payload = r["contrast_thresholds_raw"]
    assert isinstance(payload, dict)
    assert set(payload.keys()) == {"poor", "fair", "good", "excellent"}

    _assert_contract_for_contrast_result(r)


def test_normalize_eval_status_fallback_used_is_fallback():
    assert normalize_eval_status("fallback_used") == "fallback"
    assert normalize_eval_status("fallback_used_with_default") == "fallback"
