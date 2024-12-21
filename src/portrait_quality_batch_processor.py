import os
import csv
from typing import List, Dict, Optional, Union
from batch_framework.base_batch import BaseBatchProcessor
from image_loader import ImageLoader
from portrait_quality_evaluator import PortraitQualityEvaluator

class PortraitQualityBatchProcessor(BaseBatchProcessor):
    """ポートレート写真の品質評価を行うバッチ処理クラス"""

    def __init__(self, config_path: Optional[str] = None, logger: Optional[object] = None, max_workers: Optional[int] = None, date: Optional[str] = None):
        """コンストラクタ: 設定ファイル、ロガー、並列処理の最大ワーカー数を初期化します。"""
        super().__init__(config_path=config_path, max_workers=max_workers)

        # ディレクトリやファイルの設定
        self.date = date
        self.image_data: List[Dict[str, str]] = []
        self.processed_images = set()
        self.image_loader = ImageLoader(logger=self.logger)

        # 設定ファイルからバッチサイズを取得（デフォルト100）
        self.batch_size = self.config.get("batch_size", 100)

    def setup(self) -> None:
        """セットアップ処理"""
        super().setup()  # 親クラスの共通セットアップ処理

        self.logger.info("Setting up PortraitQualityBatchProcessor...")
        self._set_directories_and_files()

        # result_csv_fileが存在する場合は削除
        if os.path.exists(self.result_csv_file):
            self.logger.info(f"Removing existing result file: {self.result_csv_file}")
            os.remove(self.result_csv_file)

        self._load_processed_images()
        self.image_data = self.load_image_data()

        # BaseBatchProcessor の data 属性に image_data を設定
        self.data = self.image_data

    def _set_directories_and_files(self) -> None:
        """ディレクトリや出力ファイルパスを設定"""
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
        """既に処理済みの画像ファイルを読み込む"""
        if os.path.exists(self.processed_images_file):
            with open(self.processed_images_file, 'r') as f:
                self.processed_images = set(f.read().splitlines())

    def load_image_data(self) -> List[Dict[str, str]]:
        """CSVファイルから画像データをロード"""
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
            self.logger.warning(f"File not found: {self.image_csv_file}. Proceeding with empty data.")
        return image_data

    def _process_batch(self, batch: List[Dict[str, str]]) -> None:
        """バッチ単位で画像処理を実行"""
        results = []

        for img_info in batch:
            try:
                result = self.process_image(
                    os.path.join(self.base_directory, img_info["file_name"]),
                    img_info["orientation"],
                    img_info["bit_depth"]
                )
                if result:
                    self.logger.info(f"Processed image: {img_info['file_name']}")
                    results.append(result)
                    self._mark_as_processed(img_info["file_name"])
            except Exception as e:
                self.logger.error(f"Failed to process image {img_info['file_name']}: {e}")

        if results:
            self.save_results(results, self.result_csv_file)

    def process_image(self, file_name: str, orientation: str, bit_depth: str) -> Optional[Dict[str, Union[str, float, bool]]]:
        """単一の画像を処理し、評価を行う"""
        self.logger.info(f"Processing image: {file_name}")
        result = {
            "file_name": os.path.basename(file_name),
            "sharpness_score": None,
            "blurriness_score": None,
            "contrast_score": None,
            "noise_score": None,
            "face_detected": None,
            "face_sharpness_score": None,
            "face_contrast_score": None,
            "face_noise_score": None
        }
        try:
            image = self.image_loader.load_image(file_name, orientation, bit_depth)
            evaluator = PortraitQualityEvaluator(image)
            result.update(evaluator.evaluate())
            return result
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_name}")
        except Exception as e:
            self.logger.error(f"Error processing image {file_name}: {e}")
        return None

    def _mark_as_processed(self, file_name: str) -> None:
        """処理済みファイルを記録"""
        with open(self.processed_images_file, 'a') as f:
            f.write(f"{file_name}\n")
        self.processed_images.add(file_name)

    def save_results(self, results: List[Dict[str, Union[str, float, bool]]], file_path: str) -> None:
        """結果をCSVファイルに保存"""
        self.logger.info(f"Saving results to {file_path}")
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)

    def cleanup(self) -> None:
        """クリーンアップ処理"""
        super().cleanup()  # 親クラスの共通クリーンアップ
        self.logger.info("Cleaning up PortraitQualityBatchProcessor-specific resources.")

        # processed_images_fileを削除
        if os.path.exists(self.processed_images_file):
            self.logger.info(f"Removing processed images file: {self.processed_images_file}")
            os.remove(self.processed_images_file)

# メイン処理
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Process images with PortraitQualityBatchProcessor.")
    parser.add_argument("--config_path", type=str, default="/home/mluser/photo_insight/config/config.yaml", help="Config file path")
    parser.add_argument("--date", type=str, help="Specify the date for directory and file names.")
    args = parser.parse_args()

    processor = PortraitQualityBatchProcessor(config_path=args.config_path, date=args.date)
    processor.execute()
