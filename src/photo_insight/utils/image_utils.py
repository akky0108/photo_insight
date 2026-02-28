import cv2
import numpy as np
import random
import rawpy


class ImageUtils:
    @staticmethod
    def preprocess_image_for_mtcNN(image, size=(640, 480), augment=False):
        """
        JPEGやPNG画像用の前処理を行います。

        :param image: 入力画像 (JPEGやPNG)
        :param size: リサイズするサイズ (幅, 高さ)
        :param augment: データ拡張を有効化するかどうか
        :return: 前処理された画像（NumPy配列形式）
        """
        # 画像をRGBに変換
        image_rgb = ImageUtils._to_rgb(image)

        # データ型を確認して適切に変換
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        # データ拡張のオプション
        if augment:
            image_rgb = ImageUtils.augment_image(image_rgb)

        # リサイズ
        image_resized = cv2.resize(image_rgb, size)

        # ピクセル値を[0, 1]の範囲にスケーリング
        image_scaled = image_resized.astype(np.float32) / 255.0

        return image_scaled

    @staticmethod
    def preprocess_raw_image(raw_input, size=(640, 480), augment=False):
        """
        RAW形式の画像を処理し、MTCNNモデルに適した形式に変換します。

        :param raw_input: RAW画像ファイルのパス、またはすでに読み込まれたnumpy配列
        :param size: リサイズするサイズ (幅, 高さ)
        :param augment: データ拡張を有効化するかどうか
        :return: 前処理された画像（NumPy配列形式）
        """
        # RAWデータの読み込み
        if isinstance(raw_input, np.ndarray):
            rgb_image = raw_input
        else:
            rgb_image = ImageUtils._load_raw_image(raw_input)

        # RGB画像をfloat32に変換し、MTCNNに適した形式にする
        rgb_image = np.asarray(rgb_image, dtype=np.float32)

        # 画素値を[0, 255]に正規化
        normalized_image = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX)

        # RGBに変換 (BGRの場合)
        normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)

        # [0, 1] に正規化
        normalized_image = normalized_image / 255.0

        return normalized_image

    @staticmethod
    def _load_raw_image(raw_image_path):
        """
        RAW画像を読み込み、RGB画像に変換します。

        :param raw_image_path: RAW画像ファイルのパス
        :return: 処理されたRGB画像のnumpy配列
        """
        try:
            with rawpy.imread(raw_image_path) as raw:
                rgb_image = raw.postprocess(output_bps=16)
            return np.array(rgb_image)
        except Exception:
            raise

    @staticmethod
    def augment_image(image):
        """
        画像の回転、スケーリング、明るさ調整を行うデータ拡張。

        :param image: 元の画像
        :return: 拡張された画像
        """
        rows, cols = image.shape[:2]
        # 画像のランダム回転
        rotation_angle = random.uniform(-15, 15)
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # スケール調整
        scale_factor = random.uniform(0.9, 1.1)
        scaled_image = cv2.resize(rotated_image, None, fx=scale_factor, fy=scale_factor)

        # 明るさ調整
        brightness_factor = random.uniform(0.8, 1.2)
        hsv = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return adjusted_image

    @staticmethod
    def apply_histogram_equalization(image):
        """
        CLAHEを使用してヒストグラム均等化を行う。

        :param image: 元の画像
        :return: ヒストグラム均等化された画像
        """
        if len(image.shape) == 3:  # カラー画像の場合
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:  # グレースケール画像
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def normalize_image(image: np.ndarray, target_bit_depth=8) -> np.ndarray:
        """
        画像を平均0、標準偏差1に正規化し、指定されたビット深度に変換します。

        :param image: 読み込んだ画像の配列
        :param target_bit_depth: 戻すときのビット深度（例：8, 14, 16）
        :return: 正規化された画像
        """
        # 正規化のために16ビット、14ビットの値を浮動小数点に変換
        # max_value = image.max()
        # normalized_image = image / max_value  # 0から1の範囲に正規化

        # ヒストグラムストレッチ（最小値と最大値を0-1の範囲にストレッチ）
        stretched_image = ImageUtils._histogram_stretch(image)

        # 出力ビット深度にスケーリング
        scaled_image = ImageUtils._scale_bit_depth(stretched_image, target_bit_depth)

        return scaled_image

    @staticmethod
    def apply_noise_filter(image: np.ndarray, method: str = "median", kernel_size: int = 5) -> np.ndarray:
        """
        ノイズを軽減するためにフィルタを適用。デフォルトはメディアンフィルタ。

        :param image: 正規化された画像
        :param method: フィルタの種類 ('median', 'gaussian', 'bilateral')
        :param kernel_size: フィルタのカーネルサイズ
        :return: フィルタ処理された画像
        """
        # 指定されたフィルタを適用
        if method == "median":
            return cv2.medianBlur(image, kernel_size)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif method == "bilateral":
            return cv2.bilateralFilter(image, kernel_size, 75, 75)
        else:
            raise ValueError(f"Unsupported filter method: {method}")

    @staticmethod
    def extract_region(image, box):
        """
        画像から指定された領域を切り出します。

        :param image: 入力画像
        :param box: 切り出す領域を指定する(x, y, 幅, 高さ)
        :return: 切り出した画像領域
        """
        x, y, w, h = box
        return image[y : y + h, x : x + w]

    @staticmethod
    def scale_region(region, target_size=(300, 300)):
        """
        切り出した領域を指定されたサイズにスケーリングします。

        :param region: 切り出した領域
        :param target_size: スケーリング後のサイズ
        :return: スケーリングされた画像
        """
        return cv2.resize(region, target_size)

    @staticmethod
    def calculate_region_weight(region_box, image_shape):
        """
        領域の重みを画像全体の面積に対する比率で計算します。

        :param region_box: 領域 (x, y, 幅, 高さ)
        :param image_shape: 入力画像の形状 (高さ, 幅, チャンネル数)
        :return: 重み (0から1の範囲)
        """
        region_area = region_box[2] * region_box[3]  # 領域の面積
        image_area = image_shape[0] * image_shape[1]  # 画像全体の面積
        return region_area / image_area

    @staticmethod
    def to_grayscale(image):
        """画像をグレースケールに変換"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def convert_color_space(image, target_color_space):
        """
        画像の色空間を変換します。

        :param image: 元の画像
        :param target_color_space: 変換する色空間の指定 (例: 'HSV', 'LAB', 'YUV')
        :return: 色空間が変換された画像
        """
        if target_color_space == "HSV":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif target_color_space == "LAB":
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif target_color_space == "YUV":
            return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif target_color_space == "GRAY":
            return ImageUtils.to_grayscale(image)
        else:
            raise ValueError(f"Unsupported color space: {target_color_space}")

    @staticmethod
    def _histogram_stretch(image):
        """
        ヒストグラムストレッチを使用して、画像のダイナミックレンジを最大化。

        :param image: 正規化された画像
        :return: ストレッチされた画像
        """
        # ヒストグラムの最小値と最大値を取得
        min_val = np.min(image)
        max_val = np.max(image)

        # 最小値と最大値を0から1の範囲にストレッチ
        stretched_image = (image - min_val) / (max_val - min_val)

        return np.clip(stretched_image, 0, 1)  # 値が範囲外に出ないようにクリップ

    @staticmethod
    def _scale_bit_depth(image, target_bit_depth):
        """
        正規化された画像を指定されたビット深度にスケーリング。

        :param image: 0-1に正規化された画像
        :param target_bit_depth: 出力するビット深度（8, 14, 16）
        :return: スケーリングされた画像
        """
        if target_bit_depth == 8:
            return (image * 255).astype(np.uint8)
        elif target_bit_depth == 14:
            return (image * 16383).astype(np.uint16)
        elif target_bit_depth == 16:
            return (image * 65535).astype(np.uint16)
        else:
            raise ValueError("Unsupported target bit depth. Use 8, 14, or 16.")

    def resize_image(image, max_dimension):
        """
        画像を指定した最大サイズにリサイズします（アスペクト比を維持）。
        :param image: 入力画像 (numpy array)
        :param max_dimension: 最大サイズ (ピクセル)
        :return: リサイズされた画像 (numpy array)
        """
        h, w = image.shape[:2]
        if max(h, w) <= max_dimension:
            return image
        scale = max_dimension / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
