import numpy as np
import pytest
from evaluators.portrait_quality.portrait_quality_evaluator import PortraitQualityEvaluator
from utils.app_logger import Logger

def create_dummy_logger(tmp_path):
    return Logger(project_root=str(tmp_path), logger_name="test_logger")

def test_portrait_quality_evaluate_returns_expected_keys(tmp_path):
    # Arrange
    dummy_img = np.ones((512, 512, 3), dtype=np.uint8)
    test_logger = create_dummy_logger(tmp_path)
    evaluator = PortraitQualityEvaluator(dummy_img, logger=test_logger)

    # Act
    results = evaluator.evaluate()

    # Assert
    assert isinstance(results, dict)
    expected_keys = {
            "sharpness_score", 
            "blurriness_score", 
            "contrast_score", 
            "noise_score", 
            "local_sharpness_score", 
            "local_sharpness_std", 
            "local_contrast_score", 
            "local_contrast_std", 
            "exposure_score",
            "mean_brightness",
            "face_detected", 
            "faces",
            "face_sharpness_score", 
            "face_contrast_score", 
            "face_noise_score", 
            "face_local_sharpness_score", 
            "face_local_sharpness_std", 
            "face_local_contrast_score", 
            "face_local_contrast_std", 
            "face_exposure_score", 
            "face_mean_brightness", 
            "yaw", 
            "pitch", 
            "roll", 
            "gaze",
            "composition_rule_based_score", 
            "face_position_score", 
            "framing_score", 
            "face_direction_score",
            "eye_contact_score"
            }  # 例：実際のキーに合わせて調整
    for key in expected_keys:
        assert key in results, f"Missing key: {key}"
        assert isinstance(results[key], (int, float, bool)), f"Unexpected type for key {key}: {type(results[key])}"
