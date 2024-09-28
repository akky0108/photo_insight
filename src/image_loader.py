import os
import imageio
import rawpy
from PIL import Image
import numpy as np
from typing import Optional
from log_util import Logger

class ImageLoader:
    """
    画像をロードし、前処理を行うクラス。
    JPEG、PNG、BMP、GIF、TIFF、RAWファイルに対応。
    """

    SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    SUPPORTED_TIFF_EXTENSIONS = ['.tiff', '.tif']
    SUPPORTED_RAW_EXTENSIONS = ['.nef', '.cr2', '.arw', '.dng', '.rw2']

    def __init__(self, logger: Optional[Logger] = None):
        """
        コンストラクタ。Loggerを設定する。
        :param logger: ログ出力を行うLoggerオブジェクト
        """
        self.logger = logger if logger else Logger(logger_name='ImageLoader')

    def load_image(self, filepath: str, output_bps: int = 8, apply_exif_rotation: bool = True, orientation: int = 1) -> np.ndarray:
        """
        ファイルパスから画像をロードし、numpy配列として返す。拡張子に応じて適切なロード処理を行う。

        :param filepath: 画像ファイルのパス
        :param output_bps: RAW画像の出力ビット深度（デフォルトは8ビット）
        :param apply_exif_rotation: EXIFの回転情報を適用するかどうか
        :param orientation: EXIFから取得したOrientationの値
        :return: 画像を表すnumpy配列
        """
        ext = os.path.splitext(filepath)[-1].lower()
        try:
            # 拡張子に応じて適切なライブラリで画像をロードする
            if ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                image = self._load_with_imageio(filepath)
            elif ext in self.SUPPORTED_TIFF_EXTENSIONS:
                image = self._load_with_pil(filepath)
            elif ext in self.SUPPORTED_RAW_EXTENSIONS:
                image = self._load_with_rawpy(filepath)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            
            # 必要に応じてEXIFの回転情報を適用する
            if apply_exif_rotation:
                image = self._apply_exif_rotation(image, orientation)

            # 必要に応じてガンマ補正を行う。
            image = self._apply_gamma_correction(image, gamma=2.2)

            return image
        except Exception as e:
            self.logger.error(f"Failed to load image from {filepath}: {e}")
            raise

    def _load_with_imageio(self, filepath: str) -> np.ndarray:
        """
        imageioを使用して画像をロードする。

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
        PIL (Pillow) を使用して画像をロードする。

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
        """
        rawpyを使用してRAW画像をロードする。

        :param filepath: 画像ファイルのパス
        :return: 読み込んだ画像のnumpy配列
        """
        try:
            with rawpy.imread(filepath) as raw:
                rgb_image = raw.postprocess(
                    output_color=rawpy.ColorSpace.Adobe, 
                    output_bps=16,        # 16ビット深度で処理
                    use_camera_wb=True,   # カメラのホワイトバランスを使用
                    no_auto_bright=True,  # 自動輝度補正を無効化
                    gamma=(1, 1)          # ガンマ補正を無効化（後で適用するため）
                    )
            return np.array(rgb_image)
        except Exception as e:
            self.logger.error(f"Failed to load image with rawpy from {filepath}: {e}")
            raise

    def _apply_exif_rotation(self, image: np.ndarray, orientation: int) -> np.ndarray:
        """
        EXIFの回転情報を使用して画像を正しい向きに補正する。

        :param image: 読み込まれた画像のnumpy配列
        :param orientation: EXIFから取得したOrientationの値
        :return: 回転補正された画像のnumpy配列
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

    def _apply_gamma_correction(self, image, gamma=2.2):
        """
        ガンマ補正を適用します。
        """
        inv_gamma = 1.0 / gamma
        image = image / 65535.0  # 16ビット画像の正規化
        corrected_image = np.power(image, inv_gamma)
        return (corrected_image * 65535).astype(np.uint16)
    
    def _resize_image(self, image: np.ndarray, resize: tuple) -> np.ndarray:
        """
        画像を指定されたサイズにリサイズする。

        :param image: リサイズ対象の画像（numpy配列）
        :param resize: (width, height) のタプルで指定された新しいサイズ
        :return: リサイズされた画像のnumpy配列
        """
        try:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(resize, Image.ANTIALIAS)
            return np.array(pil_image)
        except Exception as e:
            self.logger.error(f"Failed to resize image: {e}")
            raise

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        画像を指定された角度で回転させる。

        :param image: 回転対象の画像（numpy配列）
        :param angle: 回転角度（度単位）
        :return: 回転された画像のnumpy配列
        """
        try:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.rotate(angle, expand=True)
            return np.array(pil_image)
        except Exception as e:
            self.logger.error(f"Failed to rotate image: {e}")
            raise

    def load_images_from_directory(self, directory: str, output_bps: int = 16, apply_exif_rotation: bool = True, orientation: int = 1) -> list:
        """
        指定されたディレクトリ内の全画像ファイルをロードし、リストとして返す。

        :param directory: 画像ファイルが格納されたディレクトリのパス
        :param output_bps: RAW画像の出力ビット深度（デフォルトは16ビット）
        :param apply_exif_rotation: EXIFの回転情報を適用するかどうか
        :return: 画像を表すnumpy配列のリスト
        """
        images = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                try:
                    images.append(self.load_image(filepath, output_bps=output_bps, apply_exif_rotation=apply_exif_rotation, orientation=orientation))
                except Exception as e:
                    self.logger.error(f"Failed to load image from {filepath}: {e}")
                    continue
        return images

 