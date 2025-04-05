import os
import cv2
import imageio.v3 as iio
import rawpy
from PIL import Image
import numpy as np
from typing import Optional
from log_util import Logger

class ImageLoader:
    """
    クラス概要:
    画像をロードし、前処理を行うクラスです。JPEG、PNG、BMP、GIF、TIFF、RAW形式の画像に対応しています。
    """

    SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    SUPPORTED_TIFF_EXTENSIONS = ['.tiff', '.tif']
    SUPPORTED_RAW_EXTENSIONS = ['.nef', '.cr2', '.arw', '.dng', '.rw2', '.orf']

    def __init__(self, logger: Optional[Logger] = None):
        """
        コンストラクタ:
        Loggerオブジェクトを使用して、処理ログを管理します。

        :param logger: ログ出力を行うLoggerオブジェクト (デフォルト: None)
        """
        self.logger = logger if logger else Logger(logger_name='ImageLoader')

    def load_image(self, filepath: str, output_bps: int = 8, apply_exif_rotation: bool = True, orientation: Optional[int] = None) -> np.ndarray:
        """
        画像のロード:
        :param filepath: 画像ファイルのパス
        :param output_bps: RAW画像のビット深度 (デフォルト: 8)
        :param apply_exif_rotation: EXIFの回転情報を適用するかどうか
        :param orientation: 事前に取得したEXIFのOrientation値（RAW画像のみ）。通常の画像はNoneでOK。
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
                image, raw_orientation = self._load_with_rawpy(filepath)

                # **整合性チェック**
                if orientation is not None and raw_orientation is not None and orientation != raw_orientation:
                    self.logger.warning(f"Orientation mismatch: provided={orientation}, detected={raw_orientation} for {filepath}")
                    orientation = raw_orientation  # 信頼できる方を使う（ここでは rawpy の値を採用）
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

            # EXIFの回転情報を適用（orientation=1 または None の場合はスキップ）
            if apply_exif_rotation and orientation and orientation != 1:
                image = self._apply_exif_rotation(image, orientation)

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

    def _load_with_rawpy(self, filepath: str) -> tuple[np.ndarray, Optional[int]]:
        """
        rawpyでRAW画像を読み込む。

        :param filepath: RAW画像のパス
        :return: (RGB画像, EXIF Orientation値 or None)
        """
        try:
            with rawpy.imread(filepath) as raw:
                # Orientation取得（存在しない場合はNone）
                orientation = raw.sizes.orientation if hasattr(raw.sizes, 'orientation') else None  
                
                rgb_image = raw.postprocess(
                    output_color=rawpy.ColorSpace.sRGB,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    gamma=(1, 1),
                    noise_thr=None
                )
                return rgb_image, orientation
        except Exception as e:
            self.logger.error(f"Failed to load RAW image with rawpy from {filepath}: {e}")
            raise

    def _apply_exif_rotation(self, image: np.ndarray, orientation: int) -> np.ndarray:
        """
        EXIFの回転情報を使用して画像の向きを調整します。

        :param image: 読み込まれた画像のnumpy配列
        :param orientation: EXIFから取得したOrientationの値
        :return: 調整後の画像
        """
        try:
            # numpy配列をPILイメージに変換
            img_pil = Image.fromarray(image)
            
            # EXIFの回転情報に基づいて画像を回転
            img_pil = img_pil.transpose(Image.Transpose.EXIF)
            
            # PILイメージをnumpy配列に戻す
            return np.array(img_pil)
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
            image = image.astype(np.float32) / 65535.0  # 0-1 の範囲に正規化
            image = np.power(image, 1.0 / gamma)
            image = (image * 65535).astype(np.uint16)  # uint16 に戻す
        else:
            image = image.astype(np.float32) / 255.0  # 0-1 の範囲に正規化
            image = np.power(image, 1.0 / gamma)
            image = (image * 255).astype(np.uint8)  # uint8 に戻す
        return image

    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """ノイズレベルを計算"""
        if image.shape[-1] == 3:  # カラーチャンネルがある場合
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # RGBで統一
        else:
            gray_image = image  # すでにグレースケールならそのまま
        return np.std(gray_image)  # 標準偏差をノイズ指標として使用

    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """動的ノイズ除去"""
        noise_level = self._calculate_noise_level(image)
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))

        if mean_brightness > 180:  # 明るい画像ではノイズが少ないのでhを弱める
            h = max(3, int(noise_level / 3))
        elif mean_brightness < 50:  # 暗い画像はノイズが多いので強める
            h = max(10, int(noise_level / 1.5))
        else:
            h = max(5, min(30, int(noise_level / 2)))

        return cv2.fastNlMeansDenoisingColored(image, None, h, templateWindowSize=7, searchWindowSize=21)

    def _sharpen_image(self, image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """ アンシャープマスキングを使用したシャープ化処理 """
        h, w, _ = image.shape
        sigma = max(0.5, min(2.0, (h + w) / 2000))  # 画像サイズに応じてsigmaを調整

        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        sharpened = cv2.addWeighted(image, alpha, blurred, -0.5, 0)
        return sharpened
    