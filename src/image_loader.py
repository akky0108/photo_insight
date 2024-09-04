import os
import imageio
import rawpy
from PIL import Image
import numpy as np
from log_util import Logger

class ImageLoader:
    def __init__(self, logger=None):
        """
        コンストラクタ。Loggerオブジェクトを受け取り、指定がない場合はデフォルトのLoggerを使用します。

        :param logger: ログ出力に使用するLoggerオブジェクト（省略可能）
        """
        self.logger = logger if logger else Logger(logger_name='ImageLoader').get_logger()

    def load_image(self, filepath: str) -> np.ndarray:
        """
        ファイルパスから画像をロードし、numpy配列として返します。
        ファイル拡張子に基づいて適切なライブラリを使用します。

        :param filepath: 画像ファイルのパス
        :return: 画像を表すnumpy配列
        :raises: ValueError: サポートされていないファイル拡張子の場合
                 Exception: 画像のロードに失敗した場合
        """
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
        """
        imageioを使用して画像をロードします。

        :param filepath: 画像ファイルのパス
        :return: 画像を表すnumpy配列
        :raises: Exception: 画像のロードに失敗した場合
        """
        try:
            return np.array(imageio.imread(filepath))
        except Exception as e:
            self.logger.error(f"Failed to load image with imageio from {filepath}: {e}")
            raise

    def _load_with_pil(self, filepath: str) -> np.ndarray:
        """
        PIL (Pillow) を使用して画像をロードします。

        :param filepath: 画像ファイルのパス
        :return: 画像を表すnumpy配列
        :raises: Exception: 画像のロードに失敗した場合
        """
        try:
            with Image.open(filepath) as img:
                return np.array(img)
        except Exception as e:
            self.logger.error(f"Failed to load image with PIL from {filepath}: {e}")
            raise

    def _load_with_rawpy(self, filepath: str) -> np.ndarray:
        """
        rawpyを使用してRAW画像をロードします。

        :param filepath: RAW画像ファイルのパス
        :return: 画像を表すnumpy配列
        :raises: Exception: 画像のロードに失敗した場合
        """
        try:
            with rawpy.imread(filepath) as raw:
                rgb_image = raw.postprocess()
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
        :raises: Exception: リサイズに失敗した場合
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
        :raises: Exception: 回転に失敗した場合
        """
        try:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.rotate(angle, expand=True)
            return np.array(pil_image)
        except Exception as e:
            self.logger.error(f"Failed to rotate image: {e}")
            raise

    def load_images_from_directory(self, directory: str) -> list:
        """
        指定されたディレクトリ内の全画像ファイルをロードし、リストとして返します。

        :param directory: 画像ファイルが格納されたディレクトリのパス
        :return: 画像を表すnumpy配列のリスト
        :raises: Exception: 各画像のロードに失敗した場合、エラーをログに記録し次の画像に進みます
        """
        images = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                try:
                    images.append(self.load_image(filepath))
                except Exception as e:
                    self.logger.error(f"Failed to load image from {filepath}: {e}")
                    continue
        return images
