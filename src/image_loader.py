import os
import imageio
import rawpy
from PIL import Image
import numpy as np
from log_util import Logger

class ImageLoader:
    def __init__(self, logger=None):
        self.logger = logger if logger else Logger(logger_name='ImageLoader').get_logger()

    def load_image(self, filepath: str) -> np.ndarray:
        ext = os.path.splitext(filepath)[-1].lower()
        try:
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                return self._load_with_imageio(filepath)
            elif ext in ['.tiff', '.tif']:
                return self._load_with_pil(filepath)
            elif ext in ['.nef', '.cr2', '.arw', '.dng', '.rw2']:
                return self._load_with_rawpy(filepath)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
        except Exception as e:
            self.logger.error(f"Failed to load image from {filepath}: {e}")
            raise

    def _load_with_imageio(self, filepath: str) -> np.ndarray:
        try:
            return np.array(imageio.imread(filepath))
        except Exception as e:
            self.logger.error(f"Failed to load image with imageio from {filepath}: {e}")
            raise

    def _load_with_pil(self, filepath: str) -> np.ndarray:
        try:
            with Image.open(filepath) as img:
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Failed to load image with PIL from {filepath}: {e}")
            raise

    def _load_with_rawpy(self, filepath: str) -> np.ndarray:
        try:
            with rawpy.imread(filepath) as raw:
                rgb_image = raw.postprocess()
            return np.array(rgb_image)
        except Exception as e:
            self.logger.error(f"Failed to load image with rawpy from {filepath}: {e}")
            raise

    def _resize_image(self, image: np.ndarray, resize: tuple) -> np.ndarray:
        try:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(resize, Image.ANTIALIAS)
            return np.array(pil_image)
        except Exception as e:
            self.logger.error(f"Failed to resize image: {e}")
            raise
