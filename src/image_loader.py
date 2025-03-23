import os
import cv2
import imageio
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

    def load_image(self, filepath: str, output_bps: int = 8, apply_exif_rotation: bool = True, orientation: int = 1) -> np.ndarray:
        """
        画像のロード:
        ファイルパスから画像をロードし、numpy配列として返します。
        画像の拡張子に応じて適切なライブラリを使用します。

        :param filepath: 画像ファイルのパス
        :param output_bps: RAW画像のビット深度 (デフォルト: 8)
        :param apply_exif_rotation: EXIFの回転情報を適用するかどうか
        :param orientation: EXIFから取得したOrientationの値
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
                image = self._load_with_rawpy(filepath)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            
            # EXIFの回転情報を適用（必要に応じて）
            if apply_exif_rotation:
                image = self._apply_exif_rotation(image, orientation)

            # 後処理前の画像を補完
            self.pre_image = image

            # RAW画像には追加の処理を適用
            if ext in self.SUPPORTED_RAW_EXTENSIONS:
                image = self._apply_gamma_correction(image, gamma=2.2)
                image = self._adjust_brightness(image)
                image = self._denoise_image(image)
                image = self._sharpen_image(image, alpha=1.5)

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
            return np.array(imageio.imread(filepath))
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

    def _load_with_rawpy(self, filepath: str) -> np.ndarray:
        try:
            with rawpy.imread(filepath) as raw:
                # RAW を最小限の処理で展開 (後処理前)
                rgb_image = raw.postprocess(
                    output_color=rawpy.ColorSpace.sRGB,
                    use_camera_wb=True,
                    no_auto_bright=True,
                    gamma=(1, 1),
                    noise_thr=None
                )

                return rgb_image

        except Exception as e:
            self.logger.error(f"Failed to load image with rawpy from {filepath}: {e}")
            raise

    def _apply_exif_rotation(self, image: np.ndarray, orientation: int) -> np.ndarray:
        """
        EXIFの回転情報を使用して画像の向きを調整します。

        :param image: 読み込まれた画像のnumpy配列
        :param orientation: EXIFから取得したOrientationの値
        :return: 調整後の画像
        """
        try:
            if orientation == 3:
                image = np.rot90(image, 2)  # 180度回転
            elif orientation == 6:
                image = np.rot90(image, -1)  # 270度回転
            elif orientation == 8:
                image = np.rot90(image, 1)  # 90度回転
        except Exception as e:
            self.logger.warning(f"Failed to apply EXIF rotation: {e}")
        
        return image

    def _adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        画像の輝度を自動補正します（CLAHEを使用）。

        :param image: 補正対象の画像
        :return: 輝度が補正された画像
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
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
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray_image)  # 標準偏差をノイズ指標として使用


    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像のノイズを除去します。

        :param image: ノイズ除去対象の画像
        :return: ノイズ除去後の画像
        """
        """動的ノイズ除去"""
        noise_level = self._calculate_noise_level(image)
        h = max(5, min(30, int(noise_level / 2)))  # ノイズレベルに応じてhを設定
        return cv2.fastNlMeansDenoisingColored(image, None, h, templateWindowSize=7, searchWindowSize=21)

    def _sharpen_image(self, image: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """
        画像のシャープネスを強調します。

        :param image: シャープ化対象の画像
        :return: シャープ化後の画像
        """
        """
        アンシャープマスキングを使用したシャープ化処理
        """
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(image, alpha, blurred, -0.5, 0)
        return sharpened    