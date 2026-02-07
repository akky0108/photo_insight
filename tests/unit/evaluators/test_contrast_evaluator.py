import numpy as np

from evaluators.contrast_evaluator import ContrastEvaluator


def test_contrast_evaluator_config_thresholds_affect_score():
    """
    config の discretize_thresholds_raw により score/grade が変わることを保証する。
    std が約30になる画像を作って判定を安定化する。
    """
    # 0 と 60 を半分ずHookり：平均との差が ±30 → std=30
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, :50] = 0
    img[:, 50:] = 60

    # A: 30 は fair(0.5)
    cfg_a = {
        "contrast": {
            "discretize_thresholds_raw": {
                "poor": 10.0,
                "fair": 20.0,
                "good": 40.0,
                "excellent": 80.0,
            }
        }
    }
    ev_a = ContrastEvaluator(config=cfg_a)
    r_a = ev_a.evaluate(img)

    assert r_a["contrast_eval_status"] == "ok"
    assert r_a["contrast_score"] == 0.5
    assert r_a["contrast_grade"] == "fair"
    assert r_a["contrast_raw"] is not None

    # B: 30 は good(0.75)
    cfg_b = {
        "contrast": {
            "discretize_thresholds_raw": {
                "poor": 5.0,
                "fair": 10.0,
                "good": 20.0,
                "excellent": 100.0,
            }
        }
    }
    ev_b = ContrastEvaluator(config=cfg_b)
    r_b = ev_b.evaluate(img)

    assert r_b["contrast_eval_status"] == "ok"
    assert r_b["contrast_score"] == 0.75
    assert r_b["contrast_grade"] == "good"

def test_contrast_float01_input_scaled():
    img = np.zeros((100, 100), dtype=np.float32)
    img[:, :50] = 0.0
    img[:, 50:] = 60.0 / 255.0  # float01

    ev = ContrastEvaluator()
    r = ev.evaluate(img)

    assert r["contrast_raw"] > 20

def test_contrast_thresholds_sorted():
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

