import pytest
import numpy as np
from photo_insight.image_utils.image_preprocessor import ImagePreprocessor


def create_dummy_image(height=300, width=400, channels=3):
    return np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)


def test_process_basic():
    preprocessor = ImagePreprocessor(resize_size=(224, 224))
    image = create_dummy_image()
    processed = preprocessor.process(image)

    assert processed.shape == (224, 224, 3)
    assert processed.dtype == np.float32
    assert np.all(processed >= 0.0) and np.all(processed <= 1.0)


def test_process_invalid_input_none():
    preprocessor = ImagePreprocessor()
    with pytest.raises(ValueError):
        preprocessor.process(None)


def test_process_invalid_input_shape():
    preprocessor = ImagePreprocessor()
    invalid_image = np.random.rand(224, 224)  # 2次元 → チャンネル不足
    with pytest.raises(ValueError):
        preprocessor.process(invalid_image)
