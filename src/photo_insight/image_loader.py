from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import imageio.v3 as iio
from PIL import Image
import numpy as np

from photo_insight.utils.app_logger import Logger

# rawpy is an optional dependency.
# CI / unit tests may not install it, so avoid import-time crash.
try:
    import rawpy  # type: ignore
except Exception:  # pragma: no cover
    rawpy = None  # type: ignore


class ImageLoader:
    """
    クラス概要:
    画像をロードし、前処理を行うクラスです。JPEG、PNG、BMP、GIF、TIFF、RAW形式の画像に対応しています。
    """

    SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    SUPPORTED_TIFF_EXTENSIONS = [".tiff", ".tif"]
    SUPPORTED_RAW_EXTENSIONS = [".nef", ".cr2", ".arw", ".dng", ".rw2", ".orf"]

    def __init__(self, logger: Optional[Logger] = None):
        """
        コンストラクタ:
        Loggerオブジェクトを使用して、処理ログを管理します。

        :param logger: ログ出力を行うLoggerオブジェクト (デフォルト: None)
        """
        self.logger = logger if logger else Logger(logger_name="ImageLoader")
        self.pre_image: Optional[np.ndarray] = None

    def load_image(
        self,
        filepath: str,
        output_bps: int = 8,
        apply_exif_rotation: bool = True,
        orientation: Optional[int] = None,
    ) -> np.ndarray:
        """
        画像のロード:
        :param filepath: 画像ファイルのパス
        :param output_bps: RAW画像のビット深度 (デフォルト: 8)
        :param apply_exif_rotation: EXIFの回転情報を適用するかどうか
        :param orientation: 事前に取得したEXIFのOrientation値（RAW画像のみ）。
            通常の画像はNoneでOK。
        :return: 読み込まれた画像のnumpy配列
        """
        ext = os.path.splitext(filepath)[-1].lower()

        try:
            # 拡張子に基づいて画像をロード
            if ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                image = self._load_with_imageio(filepath)
            elif ext in self.SUPPORTED_TIFF_EXTENSIONS:
                image = self._load_with_pil(filepath)
            elif ext in self.SUPPORTED_RAW_EXTENSIONS:
                image, raw_orientation = self._load_with_rawpy(filepath, output_bps=output_bps)

                # **整合性チェック**
                if (
                    orientation is not None
                    and raw_orientation is not None
                    and orientation != raw_orientation
                ):
                    self.logger.warning(
                        f"Orientation mismatch: provided={orientation}, "
                        f"detected={raw_orientation} for {filepath}"
                    )
                    # rawpy 由来を採用（rawpy が読めているのでこちらを信頼）
                    orientation = raw_orientation
                else:
                    # orientation が未指定なら rawpy の値を採用
                    if orientation is None and raw_orientation is not None:
                        orientation = raw_orientation
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

            # EXIFの回転情報を適用（orientation=1 または None の場合はスキップ）
            if apply_exif_rotation and orientation and orientation != 1:
                image = self._apply_exif_rotation(image, orientation)

            self.pre_image = image
            return image

        except Exception as e:
            self.logger.error(f"Failed to load image from {filepath}: {e}")
            raise

    def get_preimage(self):
        return self.pre_image

    def _load_with_imageio(self, filepath: str) -> np.ndarray:
        """
        imageioライブラリで画像をロードします。

        :param filepath: 画像ファイルのパス
        :return: 読み込んだ画像のnumpy配列
        """
        try:
            return np.array(iio.imread(filepath))
        except Exception as e:
            self.logger.error(f"Failed to load image with imageio from {filepath}: {e}")
            raise

    def _load_with_pil(self, filepath: str) -> np.ndarray:
        """
        PIL (Pillow) ライブラリでTIFF画像をロードします。

        :param filepath: 画像ファイルのパス
        :return: 読み込んだ画像のnumpy配列
        """
        try:
            with Image.open(filepath) as img:
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Failed to load image with PIL from {filepath}: {e}")
            raise

    def _load_with_rawpy(self, filepath: str, output_bps: int = 8) -> Tuple[np.ndarray, Optional[int]]:
        """
        rawpyでRAW画像を読み込む。

        :param filepath: RAW画像のパス
        :param output_bps: rawpy postprocess の出力ビット深度 (8 or 16)
        :return: (RGB画像, EXIF Orientation値 or None)
        """
        if rawpy is None:
            raise RuntimeError(
                "rawpy is required to load RAW images, but it is not installed. "
                "Install `rawpy` (and system deps like libraw if needed), "
                "or mock ImageLoader in unit tests/CI."
            )

        try:
            with rawpy.imread(filepath) as raw:  # type: ignore[attr-defined]
                # Orientation取得（存在しない場合はNone）
                raw_orientation = (
                    raw.sizes.orientation if hasattr(raw.sizes, "orientation") else None
                )

                # rawpy postprocess
                rgb_image = raw.postprocess(
                    output_color=rawpy.ColorSpace.sRGB,  # type: ignore[attr-defined]
                    use_camera_wb=True,
                    no_auto_bright=True,
                    gamma=(1, 1),
                    noise_thr=None,
                    output_bps=int(output_bps),
                )
                return rgb_image, raw_orientation
        except Exception as e:
            self.logger.error(f"Failed to load RAW image with rawpy from {filepath}: {e}")
            raise

    def _apply_exif_rotation(self, image: np.ndarray, orientation: int) -> np.ndarray:
        """
        EXIFの回転情報(Orientation 1-8)を使用して画像の向きを調整します。

        :param image: 読み込まれた画像のnumpy配列
        :param orientation: EXIFから取得したOrientationの値
        :return: 調整後の画像
        """
        try:
            img_pil = Image.fromarray(image)

            # EXIF Orientation mapping (1..8)
            # Reference: TIFF/EXIF orientation definitions
            if orientation == 1:
                transformed = img_pil
            elif orientation == 2:
                transformed = img_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                transformed = img_pil.transpose(Image.Transpose.ROTATE_180)
            elif orientation == 4:
                transformed = img_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                transformed = img_pil.transpose(Image.Transpose.TRANSPOSE)
            elif orientation == 6:
                transformed = img_pil.transpose(Image.Transpose.ROTATE_270)
            elif orientation == 7:
                transformed = img_pil.transpose(Image.Transpose.TRANSVERSE)
            elif orientation == 8:
                transformed = img_pil.transpose(Image.Transpose.ROTATE_90)
            else:
                # Unknown orientation => no-op
                self.logger.warning(f"Unknown EXIF orientation={orientation}. Skip rotation.")
                transformed = img_pil

            return np.array(transformed)
        except Exception as e:
            self.logger.warning(f"Failed to apply EXIF rotation: {e}")
            return image

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray_image.shape
        area = h * w  # 画像のピクセル数

        if area < 500_000:  # 小さい画像
            grid_size = (4, 4)
        elif area < 2_000_000:  # 中サイズ
            grid_size = (8, 8)
        else:  # 大きい画像
            grid_size = (16, 16)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=grid_size)
        equalized_image = clahe.apply(gray_image)
        return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2RGB)

    def _apply_gamma_correction(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """
        ガンマ補正を適用します。

        :param image: ガンマ補正対象の画像
        :param gamma: ガンマ値（デフォルトは2.2）
        :return: ガンマ補正後の画像
        """
        if image.dtype == np.uint16:
            image_f = image.astype(np.float32) / 65535.0
            image_f = np.power(image_f, 1.0 / gamma)
            return (image_f * 65535.0).astype(np.uint16)

        image_f = image.astype(np.float32) / 255.0
        image_f = np.power(image_f, 1.0 / gamma)
        return (image_f * 255.0).astype(np.uint8)

    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """ノイズレベルを計算"""
        if image.ndim == 3 and image.shape[-1] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # RGBで統一
        else:
            gray_image = image  # すでにグレースケールならそのまま
        return float(np.std(gray_image))

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """動的ノイズ除去"""
        noise_level = self._calculate_noise_level(image)
        mean_brightness = float(np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)))

        if mean_brightness > 180:
            h = max(3, int(noise_level / 3))
        elif mean_brightness < 50:
            h = max(10, int(noise_level / 1.5))
        else:
            h = max(5, min(30, int(noise_level / 2)))

        return cv2.fastNlMeansDenoisingColored(
            image, None, h, templateWindowSize=7, searchWindowSize=21
        )

    def _sharpen_image(self, image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """アンシャープマスキングを使用したシャープ化処理"""
        h, w, _ = image.shape
        sigma = max(0.5, min(2.0, (h + w) / 2000))  # 画像サイズに応じてsigmaを調整

        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, alpha, blurred, -0.5, 0)
        return sharpened
