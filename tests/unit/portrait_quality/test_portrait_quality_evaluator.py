import numpy as np
from evaluators.portrait_quality.portrait_quality_evaluator import PortraitQualityEvaluator
from utils.app_logger import Logger

def test_evaluate_basic():
    dummy_img = np.ones((512, 512, 3), dtype=np.uint8)
    test_logger = Logger(project_root="/home/mluser/photo_insight", logger_name="test_logger")
    evaluator = PortraitQualityEvaluator(dummy_img, logger=test_logger)
    results = evaluator.evaluate()
    assert isinstance(results, dict)