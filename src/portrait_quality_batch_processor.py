import os
import csv
import gc
from pathlib import Path
from typing import Any, List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from batch_framework.base_batch import BaseBatchProcessor
from image_loader import ImageLoader
from evaluators.portrait_quality.portrait_quality_evaluator import (
    PortraitQualityEvaluator,
)
from portrait_quality_header import PortraitQualityHeaderGenerator
from monitoring.memory_monitor import MemoryMonitor


class PortraitQualityBatchProcessor(BaseBatchProcessor):
    """ポートレート写真の品質評価を行うバッチ処理クラス"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        logger: Optional[object] = None,
        max_workers: Optional[int] = None,
        date: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        バッチ処理クラスの初期化。
        設定ファイルやロガー、日付指定などを受け取る。
        """
        super().__init__(config_path=config_path, max_workers=max_workers)

        self.config = self.config_manager.get_config()
        self.logger = logger or self.config_manager.get_logger("PortraitQualityBatchProcessor")
        self.memory_monitor = MemoryMonitor(self.logger)
        self.date = date
        self.processed_images = set()
        self.image_loader = ImageLoader(logger=self.logger)
 
        self.image_csv_file = None
        self.result_csv_file = None
        self.processed_images_file = None
        self.output_directory = None
        self.base_directory = None
        self.completed_all_batches = False
        self.memory_threshold_exceeded = False

        self.batch_size = batch_size or self.config.get("batch_size", 10)

    def on_config_change(self, new_config: dict) -> None:
        """
        設定ファイルが変更された際のハンドラ。
        バッチサイズや出力先ディレクトリを更新する。
        """
        self.logger.info("Config updated. Applying new settings...")
        self.batch_size = new_config.get("batch_size", self.batch_size)
        self.output_directory = new_config.get(
            "output_directory", self.output_directory
        )
        os.makedirs(self.output_directory, exist_ok=True)

    def setup(self) -> None:
        """
        バッチ処理の事前準備。
        ディレクトリやファイルパスの設定、既処理データの読み込みなどを行う。
        """
        self.logger.info("Setting up PortraitQualityBatchProcessor...")
        self._set_directories_and_files()
        super().setup()

        self._load_processed_images()

        if self.processed_images:
            self.logger.info(
                f"Found {len(self.processed_images)} previously processed images. Resuming from there."
            )
        else:
            self.logger.info("No previously processed images found. Starting fresh.")

        self.memory_threshold_exceeded = False
        self.completed_all_batches = False

        self.memory_threshold = self.config_manager.get_memory_threshold(default=90)
        self.logger.info(f"Memory usage threshold set to {self.memory_threshold}% from config.")

    def _set_directories_and_files(self) -> None:
        """
        ディレクトリとファイルパスを設定するヘルパー。
        日付指定がある場合はそれに応じたパスを生成。
        """
        self.output_directory = self.config.get("output_directory", "temp")
        os.makedirs(self.output_directory, exist_ok=True)

        base_directory_root = self.config.get(
            "base_directory_root", "/mnt/l/picture/2024"
        )
        if self.date:
            self.base_directory = os.path.join(base_directory_root, self.date)
            self.image_csv_file = os.path.join(
                self.output_directory, f"{self.date}_raw_exif_data.csv"
            )
            self.result_csv_file = os.path.join(
                self.output_directory, f"evaluation_results_{self.date}.csv"
            )
            self.processed_images_file = os.path.join(
                self.output_directory, f"processed_images_{self.date}.txt"
            )
        else:
            self.base_directory = self.config.get(
                "base_directory", "/mnt/l/picture/2024/2024-07-02/"
            )
            self.image_csv_file = os.path.join(
                self.output_directory, "nef_exif_data.csv"
            )
            self.result_csv_file = os.path.join(
                self.output_directory, "evaluation_results.csv"
            )
            self.processed_images_file = os.path.join(
                self.output_directory, "processed_images.txt"
            )

        os.makedirs(self.base_directory, exist_ok=True)
        self.logger.info(f"ベースディレクトリ: {self.base_directory}")

    def _load_processed_images(self) -> None:
        """
        既に処理済みの画像ファイル名を読み込む。
        """
        if os.path.exists(self.processed_images_file):
            with open(self.processed_images_file, "r") as f:
                self.processed_images = set(f.read().splitlines())
            self.logger.info(
                f"Loaded {len(self.processed_images)} previously processed images."
            )

        if os.path.exists(self.result_csv_file):
            self.logger.info(
                f"Result file exists: {self.result_csv_file} — continuing from previous run."
            )

    def load_image_data(self) -> List[Dict[str, str]]:
        """
        画像のメタデータ（ファイル名、向き、ビット深度）を CSV から読み込む。

        Returns:
            List[Dict[str, str]]: 画像情報のリスト
        """
        self.logger.info(f"Loading image data from {self.image_csv_file}")
        image_data = []
        try:
            with open(self.image_csv_file, "r") as file:
                reader = csv.DictReader(file)
                required_keys = {"FileName", "Orientation", "BitDepth"}

                if not required_keys.issubset(reader.fieldnames or set()):
                    self.logger.error(
                        f"Missing required columns in CSV. Found: {reader.fieldnames}"
                    )
                    return []

                for row in reader:
                    image_data.append(
                        {
                            "file_name": row["FileName"],
                            "orientation": row["Orientation"],
                            "bit_depth": row["BitDepth"],
                        }
                    )
        except FileNotFoundError:
            self.logger.warning(
                f"File not found: {self.image_csv_file}. Proceeding with empty data."
            )
        except csv.Error as e:
            self.logger.error(f"CSV parsing error in {self.image_csv_file}: {e}")
        return image_data

    def execute(self, target_dir: Optional[Path] = None) -> None:
        """
        バッチ処理の実行エントリポイント。
        """
        try:
            self.setup()
            data = self.get_data(target_dir=target_dir)
            self.process(data)
            self.completed_all_batches = True
        except Exception as e:
            self.handle_error(str(e), raise_exception=False)
        finally:
            self.cleanup()

    def _process_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        指定された画像バッチを処理する。
        処理済み画像はスキップし、結果を CSV に保存する。
        """
        batch = [img for img in batch if img["file_name"] not in self.processed_images]
        if not batch:
            self.logger.info(
                "All images in this batch are already processed. Skipping."
            )
            return

        if (self.max_workers or 2) == 1:
            results = self._process_batch_serial(batch)
        else:
            results = self._process_batch_parallel(batch)

        if results:
            self.save_results(results, self.result_csv_file)

        gc.collect()
        self.memory_monitor.log_usage(prefix="Post batch GC")

        if self.memory_monitor.get_memory_usage() > self.memory_threshold:
            self.logger.warning(
                f"Memory usage exceeded threshold ({self.memory_threshold}%). Will stop after this batch."
            )
            self.memory_threshold_exceeded = True

    def process_image(
        self, file_name: str, orientation: str, bit_depth: str
    ) -> Optional[Dict[str, Union[str, float, bool]]]:
        """
        単一画像を読み込み、評価処理を行い、結果を辞書形式で返す。

        Args:
            file_name (str): 画像ファイルパス
            orientation (str): 画像の向き
            bit_depth (str): ビット深度

        Returns:
            Optional[Dict[str, Union[str, float, bool]]]: 評価結果または None
        """
        self.logger.info(f"Processing image: {file_name}")
        result = {
            "file_name": os.path.basename(file_name),
            "sharpness_score": None,
            "blurriness_score": None,
            "contrast_score": None,
            "noise_score": None,
            "local_sharpness_score": None,
            "local_sharpness_std": None,
            "local_contrast_score": None,
            "local_contrast_std": None,
            "face_detected": None,
            "faces": None,
            "face_sharpness_score": None,
            "face_contrast_score": None,
            "face_noise_score": None,
            "face_local_sharpness_score": None,
            "face_local_sharpness_std": None,
            "face_local_contrast_score": None,
            "face_local_contrast_std": None,
        }

        try:
            image = self.image_loader.load_image(file_name, orientation, bit_depth)
            evaluator = PortraitQualityEvaluator(image, False, self.logger, file_name)
            eval_result = evaluator.evaluate()
            del image
            del evaluator

            if eval_result:
                result.update(eval_result)
                return result
            else:
                self.logger.warning(f"No evaluation result for image {file_name}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_name}")
        except Exception as e:
            self.logger.error(f"Error processing image {file_name}: {e}")
        return None

    def _process_single_image(
        self, img_info: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        try:
            result = self.process_image(
                os.path.join(self.base_directory, img_info["file_name"]),
                img_info["orientation"],
                img_info["bit_depth"],
            )
            if result:
                self.logger.info(f"Processed image: {img_info['file_name']}")
                self._mark_as_processed(img_info["file_name"])
                return result
        except Exception as e:
            self.logger.error(f"Failed to process image {img_info['file_name']}: {e}")
        return None

    def _process_batch_serial(
        self, batch: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        results = []
        for img_info in batch:
            result = self._process_single_image(img_info)
            if result:
                results.append(result)
        return results

    def _process_batch_parallel(
        self, batch: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers or 2) as executor:
            future_to_img = {
                executor.submit(self._process_single_image, img_info): img_info
                for img_info in batch
            }

            for future in as_completed(future_to_img):
                result = future.result()
                if result:
                    results.append(result)
        return results

    def _mark_as_processed(self, file_name: str) -> None:
        """
        処理済み画像として記録し、再処理を防止する。
        """
        with open(self.processed_images_file, "a") as f:
            f.write(f"{file_name}\n")
        self.processed_images.add(file_name)

    def save_results(
        self, results: List[Dict[str, Union[str, float, bool]]], file_path: str
    ) -> None:
        """
        画像評価の結果を CSV ファイルに保存する。

        Args:
            results (List[Dict]): 保存対象の評価結果
            file_path (str): 出力ファイルパス
        """
        self.logger.info(f"Saving results to {file_path}")
        file_exists = os.path.isfile(file_path)
        results = sorted(results, key=lambda x: x["file_name"])

        header_generator = PortraitQualityHeaderGenerator()
        fieldnames = header_generator.get_all_headers()

        with open(file_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, extrasaction="ignore"
            )
            if not file_exists:
                writer.writeheader()

            for row in results:
                writer.writerow({key: row.get(key, None) for key in fieldnames})

    def cleanup(self) -> None:
        """
        後処理のクリーンアップ処理。
        全件処理が完了していれば、一時ファイルを削除する。
        """
        super().cleanup()
        self.logger.info(
            "Cleaning up PortraitQualityBatchProcessor-specific resources."
        )
        processed_file = getattr(self, "processed_images_file", None)
        if getattr(self, "completed_all_batches", False) and processed_file and os.path.exists(processed_file):
            self.logger.info(f"Removing processed images file: {processed_file}")
            os.remove(processed_file)

    def get_data(self, target_dir: Optional[Path] = None) -> List[Dict[str, str]]:
        """
        処理対象の画像データのうち、未処理のものだけを返す。

        Returns:
            List[Dict[str, str]]: 未処理画像のメタデータ一覧
        """
        if target_dir:
            self.logger.warning(
                f"[get_data] target_dir={target_dir} は未対応のため無視されます。"
            )

        raw_data = self.load_image_data()
        return [d for d in raw_data if d["file_name"] not in self.processed_images]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process images with PortraitQualityBatchProcessor."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/mluser/photo_insight/config/config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--date", type=str, help="Specify the date for directory and file names."
    )
    parser.add_argument("--max_workers", type=int, help="Number of worker threads")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    args = parser.parse_args()
    print("[DEBUG] args:", args) 

    processor = PortraitQualityBatchProcessor(
        config_path=args.config_path,
        date=args.date,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )
    processor.execute()
