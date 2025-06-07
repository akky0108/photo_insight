import os
import numpy as np
import cv2
import pytest
from evaluators.portrait_quality.portrait_quality_evaluator import (
    PortraitQualityEvaluator,
)
from utils.app_logger import Logger

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets"))

@pytest.mark.parametrize(
    "filename, expected_face_detected",
    [
        ("face_sample.jpg", True),
        ("noface_sample.png", False),
    ],
)

def create_dummy_face_image(width=512, height=512):
    """顔らしき特徴を持つシンプルな画像を作成"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 顔の輪郭（楕円）
    center = (width // 2, height // 2)
    axes = (100, 130)
    cv2.ellipse(
        img,
        center,
        axes,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=(0, 0, 0),
        thickness=3,
    )

    # 目（円2つ）
    cv2.circle(img, (center[0] - 40, center[1] - 30), 10, (0, 0, 0), -1)
    cv2.circle(img, (center[0] + 40, center[1] - 30), 10, (0, 0, 0), -1)

    # 口（楕円）
    cv2.ellipse(img, (center[0], center[1] + 50), (40, 20), 0, 0, 180, (0, 0, 0), 3)

    return img

def create_dummy_logger(tmp_path):
    return Logger(project_root=str(tmp_path), logger_name="test_logger")

def test_portrait_quality_evaluate_face_keys_and_types(tmp_path):
    dummy_img = np.ones((512, 512, 3), dtype=np.uint8)
    logger = create_dummy_logger(tmp_path)
    evaluator = PortraitQualityEvaluator(dummy_img, logger=logger)

    results = evaluator.evaluate()

    assert "face_detected" in results
    assert isinstance(results["face_detected"], bool)

    assert "faces" in results
    assert isinstance(results["faces"], list)

    if results["face_detected"]:
        assert len(results["faces"]) > 0
        face = results["faces"][0]
        assert isinstance(face, dict)
        assert "box" in face or "bbox" in face

def test_portrait_quality_evaluate_no_face_detected(tmp_path):
    # 黒画像で顔検出されない前提
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    logger = create_dummy_logger(tmp_path)
    evaluator = PortraitQualityEvaluator(dummy_img, logger=logger)

    results = evaluator.evaluate()

    assert "face_detected" in results
    assert results["face_detected"] is False

    assert "faces" in results
    assert isinstance(results["faces"], list)
    assert len(results["faces"]) == 0

def test_portrait_quality_evaluation(filename, expected_face_detected):
    path = os.path.join(ASSETS_DIR, filename)
    assert os.path.exists(path), f"テスト画像が存在しません: {path}"

    evaluator = PortraitQualityEvaluator(image_input=path)
    result = evaluator.evaluate()

    assert isinstance(result, dict), "結果は辞書であるべき"
    assert "face_detected" in result, "'face_detected' キーが存在しない"
    assert (
        result["face_detected"] == expected_face_detected
    ), f"顔検出結果が期待と異なります: {filename}"

    # 主要スコアが返っていることも確認（最低限チェック）
    for key in ["sharpness_score", "blurriness_score", "contrast_score"]:
        assert key in result, f"{key} が結果に含まれていません"
