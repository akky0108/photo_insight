import numpy as np
from face_detectors.face_processor import FaceProcessor


class TestFaceProcessor:
    def test_get_best_face_returns_highest_confidence(self):
        faces = [{"confidence": 0.8}, {"confidence": 0.95}, {"confidence": 0.7}]
        best = FaceProcessor.get_best_face(faces)
        assert best["confidence"] == 0.95

    def test_get_best_face_returns_none_on_empty_list(self):
        assert FaceProcessor.get_best_face([]) is None

    def test_crop_face_valid_box(self):
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        box = [10, 20, 60, 80]
        crop = FaceProcessor.crop_face(image, box)
        assert crop.shape == (60, 50, 3)

    def test_crop_face_invalid_box_type(self):
        image = np.ones((100, 100, 3), dtype=np.uint8)
        crop = FaceProcessor.crop_face(image, None)
        assert crop is None

    def test_crop_face_out_of_bounds_box(self):
        image = np.ones((50, 50, 3), dtype=np.uint8)
        box = [40, 40, 100, 100]
        crop = FaceProcessor.crop_face(image, box)
        # shapeは10x10、超えた部分はクロップされる
        assert crop.shape == (10, 10, 3)
