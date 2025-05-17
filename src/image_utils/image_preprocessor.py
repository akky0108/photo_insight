import numpy as np
import cv2
from typing import Optional, Union
from PIL import Image, ExifTags
from image_loader import ImageLoader
from utils.image_utils import ImageUtils
from utils.app_logger import Logger


class ImagePreprocessor:
    def __init__(self, logger: Optional[Logger] = None, is_raw: bool = False, gamma: Optional[float] = None):
        self.logger = logger or Logger("ImagePreprocessor")
        self.image_loader = ImageLoader(logger=self.logger)
        self.is_raw = is_raw
        self.gamma = gamma  # None なら補正しない

    def load_and_resize(self, image_input: Union[str, np.ndarray]) -> dict:
        image = self._load_image(image_input)
        image = self._correct_orientation(image_input, image)
        image = self._convert_bgr_to_rgb(image)
        if self.gamma is not None:
            image = self._adjust_gamma(image, self.gamma)

        resized = {
            "original": image,
            "resized_2048": ImageUtils.resize_image(image, max_dimension=2048),
            "resized_1024": ImageUtils.resize_image(image, max_dimension=1024),
        }
        return resized

    def _load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, str):
            return self.image_loader.load_image(image_input, output_bps=16 if self.is_raw else 8)
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise ValueError("Invalid input type for image.")

    def _convert_bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.shape[-1] == 3:  # Assume color image
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _correct_orientation(self, image_path: Union[str, np.ndarray], image: np.ndarray) -> np.ndarray:
        if not isinstance(image_path, str):
            return image  # ndarray なら回転補正スキップ

        try:
            pil_image = Image.open(image_path)
            exif = pil_image._getexif()
            if not exif:
                return image

            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None
            )
            if orientation_key is None:
                return image

            orientation = exif.get(orientation_key, 1)

            if orientation == 3:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif orientation == 6:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        except Exception as e:
            self.logger.warning(f"EXIF 回転補正に失敗: {str(e)}")

        return image
