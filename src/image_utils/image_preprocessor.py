import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(
        self,
        resize_size=(224, 224),
        denoise: bool = False,
        adjust_contrast: bool = False,
        logger=None
    ):
        self.resize_size = resize_size
        self.denoise = denoise
        self.adjust_contrast = adjust_contrast
        self.logger = logger

        if self.logger:
            self.logger.debug(f"Resizing image to {self.resize_size}")

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None.")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input must be a numpy array, but got {type(image)}.")
        if image.ndim != 3 or image.shape[2] not in [3, 4]:
            raise ValueError(
                f"Invalid image shape: expected 3D array with 3 or 4 channels, got {image.shape}."
            )

        # RGB変換（必要であればチャンネル数に応じて調整）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.denoise:
            image = cv2.GaussianBlur(image, (3, 3), 0)
            if self.logger:
                self.logger.debug("ガウシアンブラーでノイズ除去を実施")

        if self.adjust_contrast:
            # HSVでVを強調する簡易処理（高度な手法に差し替え可能）
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            if self.logger:
                self.logger.debug("ヒストグラム均等化でコントラスト調整を実施")

        # リサイズ
        image = cv2.resize(image, self.resize_size)

        # 正規化
        image = image / 255.0
        return image.astype(np.float32)
