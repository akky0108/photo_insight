import cv2
import numpy as np
import random
import rawpy

class ImageUtils:
    @staticmethod
    def preprocess_image(image, size=(300, 300)):
        """
        JPEGやPNG画像用の前処理を行います。

        :param image: 入力画像 (JPEGやPNG)
        :param size: リサイズするサイズ (幅, 高さ)
        :return: 前処理された画像
        """
        # BGRからRGBに変換（OpenCVはBGRで画像を読み込むため）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # リサイズ
        image_resized = cv2.resize(image_rgb, size)

        # 画像をBlob形式に変換
        blob = cv2.dnn.blobFromImage(image_resized, 1.0, size, (104.0, 177.0, 123.0))

        return blob

    @staticmethod
    def preprocess_raw_image(raw_image_path, size=(300, 300)):
        """
        RAW形式の画像を処理し、DNNモデルに適した形式に変換します。

        :param raw_image_path: RAW画像ファイルのパス
        :param size: リサイズするサイズ (幅, 高さ)
        :return: 前処理された画像（Blob形式）
        """
        # 1. RAWデータを読み込み、デマトリクス処理を行ってRGB画像に変換
        rgb_image = ImageUtils.process_raw_image(raw_image_path)

        # 2. データ拡張（ランダムな回転、スケーリング、明るさ調整）
        augmented_image = ImageUtils.augment_image(rgb_image)

        # 3. ヒストグラム均等化
        equalized_image = ImageUtils.apply_histogram_equalization(augmented_image)

        # 4. BGRからRGBに変換（OpenCVはBGRで画像を読み込むため）
        image_rgb = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB)

        # 5. リサイズ
        image_resized = cv2.resize(image_rgb, size)

        # 6. 画像をBlob形式に変換
        blob = cv2.dnn.blobFromImage(image_resized, 1.0, size, (104.0, 177.0, 123.0))

        return blob

    @staticmethod
    def preprocess_raw_image(image_or_path):
        """
        RAW画像を前処理する。
        画像ファイルパスまたはnumpy配列を受け取り、前処理された画像を返す。

        :param image_or_path: 画像ファイルのパス、またはnumpy配列
        :return: 前処理された画像のnumpy配列
        """
        # image_or_path が numpy.ndarray かどうかを確認する
        if isinstance(image_or_path, np.ndarray):
            # すでに画像がロードされている場合、そのまま返す
            return image_or_path
        else:
            # ファイルパスの場合は、RAW画像として処理する
            return ImageUtils.process_raw_image(image_or_path)

    @staticmethod
    def process_raw_image(raw_image_path):
        """
        RAW画像を処理し、RGB画像として返す。

        :param raw_image_path: RAW画像ファイルのパス
        :return: 処理されたRGB画像のnumpy配列
        """
        try:
            with rawpy.imread(raw_image_path) as raw:
                rgb_image = raw.postprocess(output_bps=16)
            return np.array(rgb_image)
        except Exception as e:
            #Logger().get_logger().error(f"Error processing RAW image {raw_image_path}: {e}")
            raise

    @staticmethod
    def augment_image(image):
        """
        画像の回転、スケーリング、明るさ調整を行うデータ拡張。

        :param image: 元の画像
        :return: 拡張された画像
        """
        # 回転
        rows, cols = image.shape[:2]
        rotation_angle = random.uniform(-15, 15)  # -15度から15度のランダムな角度で回転
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # スケーリング
        scale_factor = random.uniform(0.9, 1.1)  # 0.9倍から1.1倍のランダムスケーリング
        scaled_image = cv2.resize(rotated_image, None, fx=scale_factor, fy=scale_factor)

        # 明るさの調整
        brightness_factor = random.uniform(0.8, 1.2)  # 明るさを0.8倍から1.2倍にランダム調整
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
        # 画像がカラーの場合は、YUV空間に変換してから適用
        if len(image.shape) == 3:  # カラー画像
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])  # 輝度チャンネル（Y）に適用
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:  # グレースケール画像
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def extract_region(image, box):
        """
        画像から指定された領域を切り出します。

        :param image: 入力画像
        :param box: 切り出す領域を指定する(x, y, 幅, 高さ)
        :return: 切り出した画像領域
        """
        x, y, w, h = box
        return image[y:y + h, x:x + w]

    @staticmethod
    def scale_region(region, target_size=(300, 300)):
        """
        切り出した顔領域を指定されたサイズにスケーリングします。

        :param region: 切り出した顔領域
        :param target_size: スケーリング後のサイズ
        :return: スケーリングされた画像
        """
        return cv2.resize(region, target_size)

    @staticmethod
    def calculate_region_weight(region_box, image_shape):
        """
        顔領域の重みを画像全体の面積に対する比率で計算します。

        :param region_box: 顔領域 (x, y, 幅, 高さ)
        :param image_shape: 入力画像の形状 (高さ, 幅, チャンネル数)
        :return: 顔領域の重み
        """
        x, y, w, h = region_box
        face_area = w * h
        image_area = image_shape[0] * image_shape[1]
        return (face_area / image_area) * 0.1  # デフォルトの重み0.1
