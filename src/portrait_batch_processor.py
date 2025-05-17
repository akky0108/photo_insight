import psutil
import os
import csv
import threading
import numpy as np
from typing import List, Optional, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor
from batch_framework.base_batch import BaseBatchProcessor
from image_loader import ImageLoader
from evaluators.portrait_quality.portrait_quality_evaluator import PortraitQualityEvaluator
from utils.image_utils import ImageUtils
import time
import gc

def log_memory_usage(logger):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_in_mb = mem_info.rss / (1024 * 1024)
    logger.info(f"[MEMORY] Current RSS: {rss_in_mb:.2f} MB")

# スレッドセーフなロックオブジェクト
lock = threading.Lock()

class PortraitBatchProcessor(BaseBatchProcessor):
    """ポートレート画像の評価指標を計算するバッチ処理を行うクラス"""
    
    # コンストラクタで初期化処理を行う
    def __init__(self, config_path: Optional[str] = None, max_workers: int = 2, date: Optional[str] = None):
        super().__init__(config_path)
        self.image_files: List[Tuple[str, Optional[int], Optional[int]]] = []
        self.max_workers = max_workers

        # ディレクトリやファイルパスを設定
        self._set_directories_and_files(date)

        self.loader = ImageLoader(self.logger)
        self.apply_filter = self.config.get('apply_filter', True)
        self.apply_normalization = self.config.get('apply_normalization', True)

        # 画像処理パイプラインの定義
        self.processing_pipeline = self._build_processing_pipeline()

    def _set_directories_and_files(self, date: Optional[str]):
        """処理するディレクトリとCSVファイルのパスを設定"""
        if date:
            self.base_directory = os.path.join(self.config.get('base_directory_root', '/mnt/l/picture/2024'), date)
            self.image_csv_file = f"{date}_nef_exif_data.csv"
        else:
            self.base_directory = self.config.get('base_directory', '/mnt/l/picture/2024/2024-07-02/')
            self.image_csv_file = "nef_exif_data.csv"
        self.output_directory = self.config.get('output_directory', 'temp')
        self.result_csv_file = os.path.join(self.output_directory, "evaluation_results.csv")

    def _build_processing_pipeline(self):
        """画像処理パイプラインの構築"""
        pipeline = []
        if self.apply_filter:
            pipeline.append(lambda img: ImageUtils.apply_noise_filter(img, method='median', kernel_size=5))
        if self.apply_normalization:
            pipeline.append(ImageUtils.normalize_image)
        return pipeline

    def setup(self) -> None:
        """バッチ処理のセットアップ。CSVから画像ファイル情報を読み込む"""
        self.logger.info("Portrait batch processor setup started.")
        image_csv_path = os.path.join(self.project_root, self.output_directory, self.image_csv_file)
        self.image_files = self._load_image_files_from_csv(image_csv_path)
        self.logger.info(f"Found {len(self.image_files)} image files listed in {image_csv_path}.")
        self._initialize_csv_files()

    def process(self) -> None:
        """画像ファイルを逐次処理し、評価結果をCSVに保存する"""
        self.logger.info("Processing images for portrait evaluation.")

        for image_file, orientation, bit_depth in self.image_files:
            self._process_single_image(image_file, orientation, bit_depth)


    def _process_single_image(self, image_file: str, orientation: Optional[int], bit_depth: Optional[int]) -> None:
        """単一の画像を処理し、評価を行う"""
        retry_count = 0
        max_retries = 3
        retry_delay = 2  # リトライの間の待機時間
        success = False

        self.logger.info(f"Processing image {image_file} with bit depth: {bit_depth}")
        log_memory_usage(self.logger)  # ← メモリログを追加

        while retry_count < max_retries and not success:
            try:
                image_path = os.path.join(self.base_directory, image_file)
                with lock:
                    image = self._load_image(image_path, orientation, bit_depth)

                if image is not None:
                    self._evaluate_image(image_file, image)
                    success = True
                else:
                    self.logger.warning(f"Failed to load image: {image_file}")
            except Exception as e:
                retry_count += 1
                self.logger.error(f"Error processing {image_file}: {e}, retrying {retry_count}/{max_retries}")
                time.sleep(retry_delay)  # リトライ前の待機
        if not success:
            self.logger.error(f"Failed to process {image_file} after {max_retries} retries.")

        if image is not None:
            self._evaluate_image(image_file, image)
            del image  # ← 明示的解放
            gc.collect()

    def _load_image_files_from_csv(self, csv_path: str) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """CSVから画像ファイル名、Orientation値、ビット深度を読み込む"""
        image_files = []
        try:
            with open(csv_path, mode='r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # ヘッダーをスキップ
                for row in reader:
                    if row:
                        image_file = row[0]
                        orientation = int(row[9]) if len(row) > 9 and row[9].isdigit() else None
                        bit_depth = int(row[10]) if len(row) > 10 and row[10].isdigit() else self.config.get('default_bit_depth', 16)
                        image_files.append((image_file, orientation, bit_depth))
        except Exception as e:
            self.logger.error(f"Error reading CSV file {csv_path}: {e}")
        return image_files

    def _load_image(self, file_path: str, orientation: Optional[int], output_bps: Optional[int]) -> Optional[np.ndarray]:
        """画像をロードし、フィルタおよび正規化を適用する"""
        try:
            # 14ビット画像の場合は16ビットに変換
            output_bps = 16 if output_bps == 14 else output_bps
            image = self.loader.load_image(file_path, output_bps=output_bps, orientation=orientation)
            self.logger.debug(f"Image loaded with dtype: {image.dtype} (bit depth: {output_bps})")

            # パイプラインで追加処理
            for process in self.processing_pipeline:
                image = process(image)

            return image
        except Exception as e:
            self.logger.error(f"Error loading image {file_path}: {e}")
            return None

    def _evaluate_image(self, image_file: str, image: np.ndarray) -> None:
        """画像を評価し、その結果をCSVに書き込む"""
        log_memory_usage(self.logger)
        is_raw = image_file.lower().endswith('.nef')
        evaluator = PortraitQualityEvaluator(image_path_or_array=image, is_raw=is_raw, logger=self.logger)
        evaluation_result = evaluator.evaluate()
        log_memory_usage(self.logger)

        self._append_evaluation_result(image_file, evaluation_result)

        del evaluator
        gc.collect()

    def _initialize_csv_files(self) -> None:
        """評価結果を記録するCSVファイルを初期化"""
        with open(self.result_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Image File", "Overall Score", "Sharpness", "Contrast", "Noise", 
                "Wavelet Sharpness", "Blurriness", "Face Detected", "Face Evaluation"
            ])

    def _append_evaluation_result(self, image_file: str, result: dict) -> None:
        """評価結果をCSVに追記する"""
        sharpness_score = result.get('sharpness', {}).get('score', 'N/A')
        contrast_score = result.get('contrast', {}).get('score', 'N/A')
        noise_score = result.get('noise', {}).get('score', 'N/A')
        wavelet_sharpness_score = result.get('wavelet_sharpness', {}).get('score', 'N/A')
        blurriness_score = result.get('blurriness', {}).get('score', 'N/A')

        with open(self.result_csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                image_file,
                result.get('overall_score', 'N/A'),
                sharpness_score,
                contrast_score,
                noise_score,
                wavelet_sharpness_score,
                blurriness_score,
                result.get('face_detected', 'N/A'),
                result.get('face_evaluation', 'N/A')
            ])

    def cleanup(self) -> None:
        """バッチ処理後のクリーンアップ処理。"""
        self.logger.info("クリーンアップが完了しました。")


def main():
    """コマンドライン引数を受け取り、バッチ処理を実行するメイン関数"""
    parser = argparse.ArgumentParser(description="Portrait Batch Processor")
    
    # デフォルトの日付を特定の日付に設定（例: 2024-08-26）
    default_date = "2024-08-26"
    
    parser.add_argument('--date', type=str, default=default_date, help=f"CSVファイルや入力フォルダに対応する日付を指定 (例: 2024-08-26). デフォルトは{default_date}です。")
    args = parser.parse_args()

    process = PortraitBatchProcessor(date=args.date, max_workers=1)
    process.execute()

if __name__ == "__main__":
    main()
