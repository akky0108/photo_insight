from evaluators.blurriness_evaluator import BlurrinessEvaluator


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
    assert ev._to_score_and_grade(15.0) == (0.25, "blurry")
    assert ev._to_score_and_grade(25.0) == (0.5, "fair")
    assert ev._to_score_and_grade(35.0) == (0.75, "good")
    assert ev._to_score_and_grade(45.0) == (1.0, "excellent")
