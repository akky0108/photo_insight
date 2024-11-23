import argparse
import os
import csv
from typing import List, Dict, Optional, Union
from batch_framework.base_batch import BaseBatchProcessor
from image_loader import ImageLoader
from log_util import Logger
from concurrent.futures import ThreadPoolExecutor, as_completed  # ThreadPoolExecutorに変更
import multiprocessing
from portrait_quality_evaluator import PortraitQualityEvaluator
import psutil

class PortraitQualityBatchProcessor(BaseBatchProcessor):
    """ポートレート写真の品質評価を行うバッチ処理クラス"""

    def __init__(self, config_path: Optional[str] = None, logger: Optional[Logger] = None, max_workers: Optional[int] = None, date: Optional[str] = None):
        config_path = config_path or os.path.join(os.getcwd(), "config.yaml")
        logger = logger or Logger(logger_name="PortraitQualityBatchProcessor")
        self.max_workers = max_workers or (multiprocessing.cpu_count() // 2)
        
        # メモリ使用量の制限を設定
        psutil.virtual_memory().percent / 2
        
        # 親クラスの初期化
        super().__init__(config_path, logger, self.max_workers)
        
        # ディレクトリやファイルの設定
        self._set_directories_and_files(date)
        
        # 処理する画像データの初期化
        self.image_data: List[Dict[str, str]] = []
        self.processed_images = set()
        self.image_loader = ImageLoader(logger=self.logger)
        
        # 処理件数の上限を設定
        self.batch_size = 100  # 100件ごとに処理するバッチサイズ

    def _set_directories_and_files(self, date: Optional[str]) -> None:
        self.output_directory = self.config.get('output_directory', 'temp')
        os.makedirs(self.output_directory, exist_ok=True)

        base_directory_root = self.config.get('base_directory_root', '/mnt/l/picture/2024')
        if date:
            self.base_directory = os.path.join(base_directory_root, date)
            self.image_csv_file = os.path.join(self.output_directory, f"{date}_raw_exif_data.csv")
            self.result_csv_file = os.path.join(self.output_directory, f"evaluation_results_{date}.csv")
            self.processed_images_file = os.path.join(self.output_directory, f"processed_images_{date}.txt")
        else:
            self.base_directory = self.config.get('base_directory', '/mnt/l/picture/2024/2024-07-02/')
            self.image_csv_file = os.path.join(self.output_directory, "nef_exif_data.csv")
            self.result_csv_file = os.path.join(self.output_directory, "evaluation_results.csv")
            self.processed_images_file = os.path.join(self.output_directory, "processed_images.txt")

    def load_image_data(self) -> List[Dict[str, str]]:
        self.logger.info(f"Loading image data from {self.image_csv_file}")
        image_data = []
        try:
            with open(self.image_csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    image_data.append({
                        "file_name": row["FileName"],
                        "orientation": row["Orientation"],
                        "bit_depth": row["BitDepth"]
                    })
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.image_csv_file}")
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
        return image_data

    def setup(self) -> None:
        self.logger.info("Setting up PortraitQualityBatchProcessor...")
        self._check_file_exists(self.base_directory, is_directory=True)
        self._check_file_exists(self.image_csv_file)

        self.image_data = self.load_image_data()
        if os.path.exists(self.processed_images_file):
            with open(self.processed_images_file, 'r') as f:
                self.processed_images = set(f.read().splitlines())

    def _check_file_exists(self, path: str, is_directory: bool = False) -> None:
        if (is_directory and not os.path.isdir(path)) or (not is_directory and not os.path.isfile(path)):
            error_type = "Directory" if is_directory else "File"
            self.logger.error(f"{error_type} not found: {path}")
            raise FileNotFoundError(f"{error_type} not found: {path}")

    def process(self) -> None:
        self.logger.info("Processing images in batches of 100...")

        unprocessed_images = [img for img in self.image_data if img["file_name"] not in self.processed_images]

        for i in range(0, len(unprocessed_images), self.batch_size):
            batch_data = unprocessed_images[i:i + self.batch_size]
            self.logger.info(f"Processing batch {i//self.batch_size + 1} with {len(batch_data)} images")

            results = []
            # ThreadPoolExecutor に変更
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.process_image, os.path.join(self.base_directory, img_info["file_name"]),
                                    img_info["orientation"], img_info["bit_depth"]): img_info for img_info in batch_data
                }

                for j, future in enumerate(as_completed(futures)):
                    img_info = futures[future]
                    try:
                        result = future.result()
                        if result:
                            self.logger.info(f"Processed image {j+1} in batch {i//self.batch_size + 1}: {os.path.basename(img_info['file_name'])}")
                            results.append(result)
                            with open(self.processed_images_file, 'a') as f:
                                f.write(f"{img_info['file_name']}\n")
                            self.processed_images.add(img_info["file_name"])
                    except Exception as e:
                        self.logger.error(f"Processing failed for {img_info['file_name']}: {e}")

            if results:
                self.save_results(results, self.result_csv_file)

            # 全ての画像が処理済みの場合、processed_images_fileを削除
            if len(self.processed_images) == len(self.image_data):
                self.logger.info("All images have been processed. Deleting processed images file.")
                os.remove(self.processed_images_file)
                break

    def process_image(self, file_name: str, orientation: str, bit_depth: str) -> Optional[Dict[str, Union[str, float, bool]]]:
        # 絶対パスではなくファイル名のみを使用
        image_name = os.path.basename(file_name)
        self.logger.info(f"Start processing image: {image_name}")

        result = {
            "file_name": image_name,  # ファイル名のみを保存
            "sharpness_score": None,
            "blurriness_score": None,
            "contrast_score": None,
            "face_detected": None,
            "face_sharpness_score": None,
            "face_contrast_score": None
        }

        try:
            image = self._load_and_preprocess_image(file_name, orientation, bit_depth)
            evaluator = PortraitQualityEvaluator(image)
            result.update(self._evaluate_image(evaluator, image_name))  # ここも修正
            return result
        except FileNotFoundError as fnfe:
            self.logger.error(f"File not found: {fnfe}")
        except Exception as e:
            self.logger.error(f"Error processing image {image_name}: {e}")
        return None

    def _load_and_preprocess_image(self, file_name: str, orientation: str, bit_depth: str):
        self.logger.info(f"Loading and preprocessing image: {file_name}")
        return self.image_loader.load_image(file_name, orientation, bit_depth)

    def _evaluate_image(self, evaluator: PortraitQualityEvaluator, file_name: str) -> Dict[str, Optional[Union[str, float, bool]]]:
        return evaluator.evaluate()

    def save_results(self, results: List[Dict[str, Union[str, float, bool]]], file_path: str) -> None:
        """結果をCSVファイルに保存。ファイルが存在する場合はヘッダを出力しない。"""
        self.logger.info(f"Saving results to {file_path}")
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)

    def run_task(self):
        pass
    
    def cleanup(self) -> None:
        self.logger.info("Cleaning up PortraitQualityBatchProcessor resources...")

# メイン処理のエントリーポイント
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with PortraitQualityBatchProcessor.")
    parser.add_argument("--config_path", type=str, default="/home/mluser/photo_insight/config/config.yaml", help="Config file path") 
    parser.add_argument('--date', type=str, help='Specify the date for directory and file names.')
    args = parser.parse_args()

    processor = PortraitQualityBatchProcessor(config_path=args.config_path, date=args.date)
    processor.setup()
    processor.process()
    processor.cleanup()
