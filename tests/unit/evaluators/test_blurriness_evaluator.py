from photo_insight.evaluators.blurriness_evaluator import BlurrinessEvaluator


def test_blurriness_evaluator_config_thresholds_affect_score():
    cfg = {
        "blurriness": {
            "discretize_thresholds_raw": {
                "bad": 10.0,
                "poor": 20.0,
                "fair": 30.0,
                "good": 40.0,
            }
        }
    }
    ev = BlurrinessEvaluator(logger=None, config=cfg)

    # raw は「大きいほど良い」
    assert ev._to_score_and_grade(5.0) == (0.0, "very_blurry")

    # 閾値は境界なので < で段階を上げる（raw==bad は 0.25 側）
    assert ev._to_score_and_grade(10.0) == (0.25, "blurry")
    assert ev._to_score_and_grade(15.0) == (0.25, "blurry")

    assert ev._to_score_and_grade(20.0) == (0.5, "fair")
    assert ev._to_score_and_grade(25.0) == (0.5, "fair")

    assert ev._to_score_and_grade(30.0) == (0.75, "good")
    assert ev._to_score_and_grade(35.0) == (0.75, "good")

    assert ev._to_score_and_grade(40.0) == (1.0, "excellent")
    assert ev._to_score_and_grade(45.0) == (1.0, "excellent")


def test_blurriness_evaluator_contract_keys_present():
    ev = BlurrinessEvaluator(logger=None, config={})
    base = ev._result_base()

    assert base["blurriness_raw_direction"] == "higher_is_better"
    assert base["blurriness_raw_transform"] == "identity"
    assert base["blurriness_higher_is_better"] is True


def test_blurriness_evaluator_thresholds_sorted_even_if_unsorted():
    cfg = {
        "blurriness": {
            "discretize_thresholds_raw": {
                # わざと順番を崩す
                "bad": 40.0,
                "poor": 10.0,
                "fair": 30.0,
                "good": 20.0,
            }
        }
    }
    ev = BlurrinessEvaluator(logger=None, config=cfg)

    # 単調性崩れの事故防止（sorted）
    assert (ev.t_bad, ev.t_poor, ev.t_fair, ev.t_good) == (10.0, 20.0, 30.0, 40.0)

def test_blurriness_evaluator_weights_are_normalized():
    ev = BlurrinessEvaluator(grad_weight=2.0, lap_weight=2.0, diff_weight=1.0, logger=None, config={})
    s = ev.grad_weight + ev.lap_weight + ev.diff_weight
    assert abs(s - 1.0) < 1e-6
    assert ev.grad_weight > 0 and ev.lap_weight > 0 and ev.diff_weight > 0

def test_blurriness_evaluator_feature_params_override_does_not_crash():
    cfg = {
        "blurriness": {
            "discretize_thresholds_raw": {"bad": 10.0, "poor": 20.0, "fair": 30.0, "good": 40.0},
            "feature_params": {
                "sobel_ksize": 5,
                "laplacian_ksize": 5,
                "gaussian_ksize": 7,
                "gaussian_sigma": 0.0,
                "clip_percentile": 99.9,
            },
        }
    }
    ev = BlurrinessEvaluator(logger=None, config=cfg)
    assert ev.sobel_ksize == 5
    assert ev.laplacian_ksize == 5
    assert ev.gaussian_ksize == 7
    assert abs(ev.clip_percentile - 99.9) < 1e-6
