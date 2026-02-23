"""
tests/unit/image_utils/test_image_preprocessor.py
"""

import pytest
import numpy as np

from photo_insight.image_utils.image_preprocessor import ImagePreprocessor


def create_dummy_image(height: int = 300, width: int = 400, channels: int = 3) -> np.ndarray:
    return np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)


def test_run_basic_resized_224_u8():
    preprocessor = ImagePreprocessor()
    image = create_dummy_image()

    out = preprocessor.run(image, max_sizes=(224,), return_uint8=True)

    assert "resized_224_u8" in out
    processed = out["resized_224_u8"]

    # aspect ratio preserved: max dimension should be 224
    assert processed.ndim == 3
    assert processed.shape[-1] == 3
    assert max(processed.shape[0], processed.shape[1]) == 224

    assert processed.dtype == np.uint8
    assert processed.min() >= 0 and processed.max() <= 255


def test_run_invalid_input_none():
    preprocessor = ImagePreprocessor()
    with pytest.raises(ValueError):
        preprocessor.run(None)  # type: ignore[arg-type]


def test_run_grayscale_input_converted_to_3ch():
    preprocessor = ImagePreprocessor()

    gray = np.random.randint(0, 256, size=(224, 224), dtype=np.uint8)

    out = preprocessor.run(gray, max_sizes=(224,), return_uint8=True)
    processed = out["resized_224_u8"]

    assert processed.ndim == 3
    assert processed.shape[-1] == 3
    assert processed.dtype == np.uint8
