import numpy as np
import cv2
from typing import Optional, Union
from PIL import Image, ExifTags
from image_loader import ImageLoader
from utils.image_utils import ImageUtils
from utils.app_logger import Logger


class ImagePreprocessor:
    def __init__(
        self,
        logger: Optional[Logger] = None,
        is_raw: bool = False,
        gamma: Optional[float] = None,
        assume_ndarray_bgr: bool = True
    ):
        self.logger = logger or Logger("ImagePreprocessor")
        self.image_loader = ImageLoader(logger=self.logger)
        self.is_raw = is_raw
        self.gamma = gamma  # None なら補正しない
        self.assume_ndarray_bgr = assume_ndarray_bgr

    def load_and_resize(self, image_input: Union[str, np.ndarray]) -> dict:
        """
        Returns:
          - RGB（従来互換）: original, resized_2048, resized_1024
          - BGR（追加）    : original_bgr, resized_2048_bgr, resized_1024_bgr
          - 評価用 uint8   : original_u8, resized_2048_u8, resized_1024_u8
                           original_bgr_u8, resized_2048_bgr_u8, resized_1024_bgr_u8
        """
        bgr = self._load_image(image_input)
        bgr = self._correct_orientation(image_input, bgr)

        if self.gamma is not None:
            bgr = self._adjust_gamma(bgr, self.gamma)

        rgb = self._convert_bgr_to_rgb(bgr)

        rgb_2048 = ImageUtils.resize_image(rgb, max_dimension=2048)
        rgb_1024 = ImageUtils.resize_image(rgb, max_dimension=1024)

        bgr_2048 = ImageUtils.resize_image(bgr, max_dimension=2048)
        bgr_1024 = ImageUtils.resize_image(bgr, max_dimension=1024)

        resized = {
            # 従来互換（RGB）
            "original": rgb,
            "resized_2048": rgb_2048,
            "resized_1024": rgb_1024,

            # 追加（BGR）
            "original_bgr": bgr,
            "resized_2048_bgr": bgr_2048,
            "resized_1024_bgr": bgr_1024,

            # ★評価用（uint8）
            "original_u8": self._to_uint8(rgb),
            "resized_2048_u8": self._to_uint8(rgb_2048),
            "resized_1024_u8": self._to_uint8(rgb_1024),

            "original_bgr_u8": self._to_uint8(bgr),
            "resized_2048_bgr_u8": self._to_uint8(bgr_2048),
            "resized_1024_bgr_u8": self._to_uint8(bgr_1024),
        }
        return resized

    def _load_image(self, image_input: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_input, str):
            return self.image_loader.load_image(
                image_input, output_bps=16 if self.is_raw else 8
            )
        elif isinstance(image_input, np.ndarray):
            if not self.assume_ndarray_bgr and image_input.ndim == 3 and image_input.shape[-1] == 3:
                # ndarrayはRGBとして渡される契約のときだけ変換
                return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
            return image_input
        else:
            raise ValueError("Invalid input type for image.")

    def _convert_bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[-1] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        if gamma is None:
            return image

        img = image.astype(np.float32)

        if image.dtype == np.uint16:
            maxv = 65535.0
            img = img / maxv
        elif image.dtype == np.uint8:
            maxv = 255.0
            img = img / maxv
        else:
            # floatは 0..1 想定に固定（max推定しない）
            maxv = 1.0
            img = np.clip(img, 0.0, 1.0)

        inv_gamma = 1.0 / float(gamma)
        img = np.clip(img, 0.0, 1.0) ** inv_gamma
        img = img * maxv

        return img.astype(image.dtype)

    def _correct_orientation(
        self, image_path: Union[str, np.ndarray], image: np.ndarray
    ) -> np.ndarray:
        if not isinstance(image_path, str):
            return image  # ndarray なら回転補正スキップ

        try:
            pil_image = Image.open(image_path)
            exif = pil_image._getexif()
            if not exif:
                return image

            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
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

    def _to_uint8(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return image
        if image.dtype == np.uint8:
            return image
        if image.dtype == np.uint16:
            # 0..65535 → 0..255（ほぼ /256、丸め誤差を抑えるなら /257）
            return (image / 257).astype(np.uint8)

        # float / int32 など
        x = image.astype(np.float32)
        # 0..1 想定なら 0..255 へ
        if x.max() <= 1.5:
            x *= 255.0
        return np.clip(x, 0, 255).astype(np.uint8)

