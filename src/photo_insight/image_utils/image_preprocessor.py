"""
src/photo_insight/image_utils/image_preprocessor.py
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import ExifTags, Image

from photo_insight.image_loader import ImageLoader
from photo_insight.utils.app_logger import Logger
from photo_insight.utils.image_utils import ImageUtils


class ImagePreprocessor:
    """Image preprocessing utility."""

    def __init__(
        self,
        logger: Optional[Logger] = None,
        is_raw: bool = False,
        gamma: Optional[float] = None,
        assume_ndarray_bgr: bool = True,
        resize_size: Optional[int] = None,  # legacy compatibility
    ):
        self.logger = logger or Logger("ImagePreprocessor")
        self.image_loader = ImageLoader(logger=self.logger)
        self.is_raw = is_raw
        self.gamma = gamma
        self.assume_ndarray_bgr = assume_ndarray_bgr
        self._legacy_resize_size = resize_size

    def run(
        self,
        image_input: Union[str, np.ndarray],
        *,
        max_sizes: Tuple[int, ...] = (2048, 1024),
        return_uint8: bool = True,
        apply_exif_orientation: bool = True,
    ) -> dict:
        """Preprocess image and return resized variants."""
        self._validate_input(image_input)

        bgr = self._load_image(image_input)
        bgr = self._ensure_3ch_bgr(bgr)  # ★ gray -> 3ch(BGR) normalize

        if apply_exif_orientation:
            bgr = self._correct_orientation(image_input, bgr)

        if self.gamma is not None:
            bgr = self._adjust_gamma(bgr, self.gamma)

        rgb = self._convert_bgr_to_rgb(bgr)

        resized: dict = {
            "original": rgb,
            "original_bgr": bgr,
        }

        for s in max_sizes:
            resized[f"resized_{s}"] = ImageUtils.resize_image(rgb, max_dimension=s)
            resized[f"resized_{s}_bgr"] = ImageUtils.resize_image(bgr, max_dimension=s)

        if return_uint8:
            resized["original_u8"] = self._to_uint8(resized["original"])
            resized["original_bgr_u8"] = self._to_uint8(resized["original_bgr"])
            for s in max_sizes:
                resized[f"resized_{s}_u8"] = self._to_uint8(resized[f"resized_{s}"])
                resized[f"resized_{s}_bgr_u8"] = self._to_uint8(resized[f"resized_{s}_bgr"])

        return resized

    def process(self, image_input: Union[str, np.ndarray]) -> dict:
        """Legacy alias for run()."""
        warnings.warn(
            "ImagePreprocessor.process() is deprecated; use run()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run(image_input)

    def load_and_resize(self, image_input: Union[str, np.ndarray]) -> dict:
        """Legacy alias for run()."""
        warnings.warn(
            "ImagePreprocessor.load_and_resize() is deprecated; use run()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.run(image_input)

    def _validate_input(self, image_input: Union[str, np.ndarray]) -> None:
        if image_input is None:
            raise ValueError("image_input must not be None")

        if isinstance(image_input, str):
            if not image_input:
                raise ValueError("image_input path must not be empty")
            return

        if isinstance(image_input, np.ndarray):
            # allow grayscale (H,W) OR color (H,W,3)
            if image_input.ndim == 2:
                return
            if image_input.ndim == 3 and image_input.shape[-1] == 3:
                return
            raise ValueError("Invalid ndarray shape for image. expected 2D or HxWx3.")

        raise ValueError("Invalid input type for image.")

    def _load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, str):
            return self.image_loader.load_image(image_input, output_bps=16 if self.is_raw else 8)

        # ndarray path:
        # - by default assume ndarray is BGR (OpenCV default)
        # - if contract says ndarray is RGB, convert to BGR
        if not self.assume_ndarray_bgr and image_input.ndim == 3:
            return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)

        return image_input

    def _ensure_3ch_bgr(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[-1] == 3:
            return img
        raise ValueError("Invalid image shape. expected 2D or HxWx3.")

    def _convert_bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        img = image.astype(np.float32)

        if image.dtype == np.uint16:
            maxv = 65535.0
            img /= maxv
        elif image.dtype == np.uint8:
            maxv = 255.0
            img /= maxv
        else:
            maxv = 1.0
            img = np.clip(img, 0.0, 1.0)

        img = np.clip(img, 0.0, 1.0) ** (1.0 / gamma)
        img *= maxv

        return img.astype(image.dtype)

    def _correct_orientation(self, image_path: Union[str, np.ndarray], image: np.ndarray) -> np.ndarray:
        # for ndarray input, skip EXIF correction
        if not isinstance(image_path, str):
            return image

        try:
            pil_image = Image.open(image_path)
            exif = pil_image._getexif()
            if not exif:
                return image

            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
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
            self.logger.warning(f"EXIF orientation correction failed: {e}")

        return image

    def _to_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image

        if image.dtype == np.uint16:
            return (image / 257).astype(np.uint8)

        x = image.astype(np.float32)

        if x.size > 0 and x.max() <= 1.5:
            x *= 255.0

        return np.clip(x, 0, 255).astype(np.uint8)
