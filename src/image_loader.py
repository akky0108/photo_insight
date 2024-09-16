import os
import imageio
import rawpy
from PIL import Image
import numpy as np
from typing import Optional
from log_util import Logger

class ImageLoader:
    """
    画像をロードし、前処理を行うクラス。JPEG、PNG、BMP、GIF、TIFF、RAWに対応。
    """

    SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    SUPPORTED_TIFF_EXTENSIONS = ['.tiff', '.tif']
    SUPPORTED_RAW_EXTENSIONS = ['.nef', '.cr2', '.arw', '.dng', '.rw2']

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger if logger else Logger(logger_name='ImageLoader')

    def load_image(self, filepath: str, output_bps: int = 8, apply_exif_rotation: bool = True) -> np.ndarray:
        """
        ファイルパスから画像をロードし、numpy配列として返す。拡張子に応じて適切なロード処理を行う。

        :param filepath: 画像ファイルのパス
        :param output_bps: RAW画像の出力ビット深度（デフォルトは16ビット）
        :param apply_exif_rotation: EXIFの回転情報を適用するかどうか
        :return: 画像を表すnumpy配列
        """
        ext = os.path.splitext(filepath)[-1].lower()
        try:
            # 拡張子に応じた処理
            if ext in self.SUPPORTED_IMAGE_EXTENSIONS:
                image = self._load_with_imageio(filepath)
            elif ext in self.SUPPORTED_TIFF_EXTENSIONS:
                image = self._load_with_pil(filepath)
            elif ext in self.SUPPORTED_RAW_EXTENSIONS:
                image = self._load_with_rawpy(filepath, output_bps=output_bps)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            
            # EXIF情報を適用する
            if apply_exif_rotation:
                image = self._apply_exif_rotation(filepath, image)

            return image
        except Exception as e:
            self.logger.error(f"Failed to load image from {filepath}: {e}")
            raise

    def _apply_exif_rotation(self, filepath: str, image: np.ndarray) -> np.ndarray:
        """
        EXIFの回転情報を使用して画像を正しい向きに補正する。

        :param filepath: 画像ファイルのパス
        :param image: 読み込まれた画像のnumpy配列
        :return: 回転補正された画像のnumpy配列
        """
        # PILでEXIF情報を確認して、回転補正を行う
        try:
            with Image.open(filepath) as img:
                exif = img._getexif()
                if exif:
                    orientation = exif.get(274)  # OrientationタグのIDは274
                    if orientation == 3:
                        image = np.rot90(image, 2)  # 180度回転
                    elif orientation == 6:
                        image = np.rot90(image, -1)  # 270度回転
                    elif orientation == 8:
                        image = np.rot90(image, 1)  # 90度回転
        except Exception as e:
            self.logger.warning(f"Failed to apply EXIF rotation to image {filepath}: {e}")
        
        return image

    def _load_with_imageio(self, filepath: str) -> np.ndarray:
        """imageioを使用して画像をロードします。"""
        try:
            return np.array(imageio.imread(filepath))
        except Exception as e:
            self.logger.error(f"Failed to load image with imageio from {filepath}: {e}")
            raise

    def _load_with_pil(self, filepath: str) -> np.ndarray:
        """PIL (Pillow) を使用して画像をロードします。"""
        try:
            with Image.open(filepath) as img:
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Failed to load image with PIL from {filepath}: {e}")
            raise

    def _load_with_rawpy(self, filepath: str, output_bps: int) -> np.ndarray:
        """rawpyを使用してRAW画像をロードします。"""
        try:
            with rawpy.imread(filepath) as raw:
                rgb_image = raw.postprocess(output_bps=output_bps)
            return np.array(rgb_image)
        except Exception as e:
            self.logger.error(f"Failed to load image with rawpy from {filepath}: {e}")
            raise

    def _resize_image(self, image: np.ndarray, resize: tuple) -> np.ndarray:
        """
        画像を指定されたサイズにリサイズします。

        :param image: リサイズ対象の画像（numpy配列）
        :param resize: (width, height) のタプルで指定された新しいサイズ
        :return: リサイズされた画像（numpy配列）
        :raises Exception: リサイズに失敗した場合
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
        画像を指定された角度で回転させます。

        :param image: 回転対象の画像（numpy配列）
        :param angle: 回転角度（度単位）
        :return: 回転された画像（numpy配列）
        :raises Exception: 回転に失敗した場合
        """
        try:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.rotate(angle, expand=True)
            return np.array(pil_image)
        except Exception as e:
            self.logger.error(f"Failed to rotate image: {e}")
            raise

    def load_images_from_directory(self, directory: str, output_bps: int = 16, apply_exif_rotation: bool = True) -> list:
        """
        指定されたディレクトリ内の全画像ファイルをロードし、リストとして返します。

        :param directory: 画像ファイルが格納されたディレクトリのパス
        :param output_bps: RAW画像の出力ビット深度（デフォルトは16ビット）
        :param apply_exif_rotation: EXIFの回転情報を適用するかどうか
        :return: 画像を表すnumpy配列のリスト
        :raises Exception: 各画像のロードに失敗した場合、エラーをログに記録し次の画像に進みます
        """
        images = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                try:
                    images.append(self.load_image(filepath, output_bps=output_bps, apply_exif_rotation=apply_exif_rotation))
                except Exception as e:
                    self.logger.error(f"Failed to load image from {filepath}: {e}")
                    continue
        return images
