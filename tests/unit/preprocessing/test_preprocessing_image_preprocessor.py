import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from photo_insight.image_utils.image_preprocessor import ImagePreprocessor


@pytest.fixture
def dummy_image():
    # 512x512 の RGB 画像
    return np.ones((512, 512, 3), dtype=np.uint8) * 128


def test_ndarray_input_without_gamma(dummy_image):
    preprocessor = ImagePreprocessor(gamma=None)
    result = preprocessor.load_and_resize(dummy_image)

    assert "original" in result
    assert result["original"].shape == dummy_image.shape
    assert result["resized_2048"].shape[0] <= 2048
    assert result["resized_1024"].shape[0] <= 1024


def test_ndarray_input_with_gamma(dummy_image):
    preprocessor = ImagePreprocessor(gamma=2.2)
    result = preprocessor.load_and_resize(dummy_image)

    assert result["original"].shape == dummy_image.shape
    # 画素値が変わっている（ガンマ補正された）
    assert not np.array_equal(result["original"], dummy_image)


@patch("photo_insight.image_utils.image_preprocessor.ImageLoader")
def test_path_input_with_exif(mock_loader_class, dummy_image, tmp_path):
    # 仮のJPEGファイル作成
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), dummy_image[:, :, ::-1])  # RGB→BGRで保存

    # ImageLoader.load_image が dummy_image を返すようにする
    mock_loader = MagicMock()
    mock_loader.load_image.return_value = dummy_image.copy()
    mock_loader_class.return_value = mock_loader

    preprocessor = ImagePreprocessor()
    result = preprocessor.load_and_resize(str(img_path))

    assert result["original"].shape == dummy_image.shape
    assert result["resized_2048"].shape[0] <= 2048
    assert result["resized_1024"].shape[0] <= 1024


def test_invalid_input_type_raises_value_error():
    preprocessor = ImagePreprocessor()
    with pytest.raises(ValueError):
        preprocessor.load_and_resize(123)  # intは不正


def test_adjust_gamma_with_zero_raises_exception():
    preprocessor = ImagePreprocessor(gamma=0)
    dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    with pytest.raises(ZeroDivisionError):
        preprocessor._adjust_gamma(dummy_image, 0)


def test_non_image_file_path_does_not_crash(tmp_path):
    dummy_path = tmp_path / "not_an_image.txt"
    dummy_path.write_text("this is not an image")
    preprocessor = ImagePreprocessor()
    # 例外にはならず警告ログが出て終わる（回転処理スキップされる）
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8)
    result = preprocessor._correct_orientation(str(dummy_path), dummy_img)
    assert isinstance(result, np.ndarray)


def test_invalid_exif_data_handled_gracefully(tmp_path):
    # 空のJPEGファイルを作る
    image_path = tmp_path / "bad_exif.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")  # JPEGのSOIとEOIだけ
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8)
    preprocessor = ImagePreprocessor()
    # 警告ログを出して処理が継続される
    result = preprocessor._correct_orientation(str(image_path), dummy_img)
    assert isinstance(result, np.ndarray)
