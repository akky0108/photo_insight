import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pytest
import numpy as np
from unittest.mock import MagicMock
from evaluators.portrait_quality.portrait_quality_evaluator import PortraitQualityEvaluator

def test_portrait_quality_evaluator():
    # モックモデルの作成
    mock_model = MagicMock()
    mock_model.predict.return_value = 0.85  # 直接値を返す

    # ダミー画像（224x224のRGB）
    dummy_image = np.random.rand(224, 224, 3).astype(np.float32)

    # PortraitQualityEvaluatorを初期化
    evaluator = PortraitQualityEvaluator(image_input=dummy_image)
    evaluator.model = mock_model  # モデルを後から設定する方法

    # 評価の実行
    result = evaluator.evaluate(dummy_image)

    # 結果は辞書型で、各スコアは数値型であるべき
    assert isinstance(result, dict)  # result が辞書型であることを確認

    # 各スコアが数値型であることを確認
    assert isinstance(result['blurriness_score'], (float, int))
    assert isinstance(result['color_balance_score'], (float, int))