import os
import csv
import gc
from typing import List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from batch_framework.base_batch import BaseBatchProcessor
from image_loader import ImageLoader
from portrait_quality_evaluator import PortraitQualityEvaluator
from portrait_quality_header import PortraitQualityHeaderGenerator
from monitoring.memory_monitor import MemoryMonitor

class PortraitQualityBatchProcessor(BaseBatchProcessor):
    """ポートレート写真の品質評価を行うバッチ処理クラス"""

    def __init__(self, config_path: Optional[str] = None, logger: Optional[object] = None,
                 max_workers: Optional[int] = None, date: Optional[str] = None,
                 batch_size: Optional[int] = None):
        super().__init__(config_path=config_path, max_workers=max_workers)

        self.logger = logger or self._get_default_logger()
        self.memory_monitor = MemoryMonitor(self.logger)
        self.date = date
        self.image_data: List[Dict[str, str]] = []
        self.processed_images = set()
        self.image_loader = ImageLoader(logger=self.logger)

        self.batch_size = batch_size or self.config.get("batch_size", 10)
        self.memory_threshold_exceeded = False  # メモリ閾値超過フラグ

    def on_config_change(self, new_config: dict) -> None:
        """設定ファイル変更検知時のハンドラ"""
        self.logger.info("Config updated. Applying new settings...")
        self.batch_size = new_config.get("batch_size", self.batch_size)
        self.output_directory = new_config.get("output_directory", self.output_directory)
        os.makedirs(self.output_directory, exist_ok=True)

    def setup(self) -> None:
        super().setup()
        self.logger.info("Setting up PortraitQualityBatchProcessor...")
        self._set_directories_and_files()

        if os.path.exists(self.result_csv_file):
            self.logger.info(f"Result file exists: {self.result_csv_file} — continuing from previous run.")

        self._load_processed_images()
        self.image_data = self.load_image_data()
        self.data = self.image_data

        if self.processed_images:
            self.logger.info(f"Found {len(self.processed_images)} previously processed images. Resuming from there.")
        else:
            self.logger.info("No previously processed images found. Starting fresh.")

        # 初期化フラグ
        self.memory_threshold_exceeded = False
        self.completed_all_batches = False

    def _set_directories_and_files(self) -> None:
        self.output_directory = self.config.get('output_directory', 'temp')
        os.makedirs(self.output_directory, exist_ok=True)

        base_directory_root = self.config.get('base_directory_root', '/mnt/l/picture/2024')
        if self.date:
            self.base_directory = os.path.join(base_directory_root, self.date)
            self.image_csv_file = os.path.join(self.output_directory, f"{self.date}_raw_exif_data.csv")
            self.result_csv_file = os.path.join(self.output_directory, f"evaluation_results_{self.date}.csv")
            self.processed_images_file = os.path.join(self.output_directory, f"processed_images_{self.date}.txt")
        else:
            self.base_directory = self.config.get('base_directory', '/mnt/l/picture/2024/2024-07-02/')
            self.image_csv_file = os.path.join(self.output_directory, "nef_exif_data.csv")
            self.result_csv_file = os.path.join(self.output_directory, "evaluation_results.csv")
            self.processed_images_file = os.path.join(self.output_directory, "processed_images.txt")

    def _load_processed_images(self) -> None:
        if os.path.exists(self.processed_images_file):
            with open(self.processed_images_file, 'r') as f:
                self.processed_images = set(f.read().splitlines())

    def load_image_data(self) -> List[Dict[str, str]]:
        self.logger.info(f"Loading image data from {self.image_csv_file}")
        image_data = []
        try:
            with open(self.image_csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if all(k in row for k in ["FileName", "Orientation", "BitDepth"]):
                        image_data.append({
                            "file_name": row["FileName"],
                            "orientation": row["Orientation"],
                            "bit_depth": row["BitDepth"]
                        })
                    else:
                        self.logger.warning(f"Skipping invalid row: {row}")
        except FileNotFoundError:
            self.logger.warning(f"File not found: {self.image_csv_file}. Proceeding with empty data.")
        return image_data

    def _process_batch(self, batch: List[Dict[str, str]]) -> None:
        batch = [img for img in batch if img["file_name"] not in self.processed_images]
        if not batch:
            self.logger.info("All images in this batch are already processed. Skipping.")
            return

        results = []
        max_workers = self.max_workers or 2

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_img = {
                executor.submit(
                    self.process_image,
                    os.path.join(self.base_directory, img_info["file_name"]),
                    img_info["orientation"],
                    img_info["bit_depth"]
                ): img_info
                for img_info in batch
            }

            for future in as_completed(future_to_img):
                img_info = future_to_img[future]
                try:
                    result = future.result()
                    if result:
                        self.logger.info(f"Processed image: {img_info['file_name']}")
                        results.append(result)
                        self._mark_as_processed(img_info["file_name"])
                except Exception as e:
                    self.logger.error(f"Failed to process image {img_info['file_name']}: {e}")

        if results:
            self.save_results(results, self.result_csv_file)

        results.clear()
        gc.collect()
        self.memory_monitor.log_usage(prefix="Post batch GC")

        # メモリ使用率チェック（閾値: 90%）
        if self.memory_monitor.get_memory_usage() > 90:
            self.logger.warning("Memory usage exceeded threshold. Will stop after this batch.")
            self.memory_threshold_exceeded = True

    def process_image(self, file_name: str, orientation: str, bit_depth: str) -> Optional[Dict[str, Union[str, float, bool]]]:
        self.logger.info(f"Processing image: {file_name}")
        result = {
            "file_name": os.path.basename(file_name),
            "sharpness_score": None,
            "blurriness_score": None,
            "contrast_score": None,
            "noise_score": None,
            'local_sharpness_score': None,
            'local_sharpness_std': None,
            'local_contrast_score': None,
            'local_contrast_std': None,
            "face_detected": None,
            "face_sharpness_score": None,
            "face_contrast_score": None,
            "face_noise_score": None,
            'face_local_sharpness_score': None,
            'face_local_sharpness_std': None,
            'face_local_contrast_score': None,
            'face_local_contrast_std': None
        }

        try:
            image = self.image_loader.load_image(file_name, orientation, bit_depth)
            evaluator = PortraitQualityEvaluator(image, False, self.logger, file_name)
            eval_result = evaluator.evaluate()
            del image
            del evaluator
            gc.collect()
            result.update(eval_result)
            return result
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_name}")
        except Exception as e:
            self.logger.error(f"Error processing image {file_name}: {e}")
        return None

    def _mark_as_processed(self, file_name: str) -> None:
        with open(self.processed_images_file, 'a') as f:
            f.write(f"{file_name}\n")
        self.processed_images.add(file_name)

    def save_results(self, results: List[Dict[str, Union[str, float, bool]]], file_path: str) -> None:
        self.logger.info(f"Saving results to {file_path}")
        file_exists = os.path.isfile(file_path)
        results = sorted(results, key=lambda x: x["file_name"])

        # PortraitQualityHeaderGenerator を使って固定の fieldnames を取得
        header_generator = PortraitQualityHeaderGenerator()
        fieldnames = header_generator.get_all_headers()

        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()

            for row in results:
                writer.writerow({key: row.get(key, None) for key in fieldnames})

    def cleanup(self) -> None:
        super().cleanup()
        self.logger.info("Cleaning up PortraitQualityBatchProcessor-specific resources.")

        # 全件完了した場合のみ途中経過ファイル削除
        if self.completed_all_batches and os.path.exists(self.processed_images_file):
            self.logger.info(f"Removing processed images file: {self.processed_images_file}")
            os.remove(self.processed_images_file)

    def execute(self):
        self.setup()
        self.logger.info("Starting image processing...")
        self.logger.info(f"Total images to process: {len(self.data)}")

        for i in range(0, len(self.data), self.batch_size):
            if self.memory_threshold_exceeded:
                self.logger.warning("Memory threshold exceeded. Halting further batch processing.")
                break

            batch = self.data[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i // self.batch_size + 1}")
            self._process_batch(batch)

        # 処理完了判定（中断再開含め対応）
        if len(self.processed_images) >= len(self.data):
            self.logger.info("All images processed successfully.")
            self.completed_all_batches = True
        else:
            self.logger.info("Some images remain unprocessed.")

        self.cleanup()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Process images with PortraitQualityBatchProcessor.")
    parser.add_argument("--config_path", type=str, default="/home/mluser/photo_insight/config/config.yaml", help="Config file path")
    parser.add_argument("--date", type=str, help="Specify the date for directory and file names.")
    parser.add_argument("--max_workers", type=int, help="Number of worker threads")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    args = parser.parse_args()

    processor = PortraitQualityBatchProcessor(
        config_path=args.config_path,
        date=args.date,
        max_workers=args.max_workers,
        batch_size=args.batch_size)
    processor.execute()
