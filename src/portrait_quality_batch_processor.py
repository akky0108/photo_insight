import argparse
import os
import csv
import numpy as np
from typing import List, Dict, Optional, Union
from batch_framework.base_batch import BaseBatchProcessor
from image_loader import ImageLoader
from log_util import Logger
from multiprocessing import Pool
import multiprocessing

from portrait_quality_evaluator import PortraitQualityEvaluator

class PortraitQualityBatchProcessor(BaseBatchProcessor):
    """ポートレート写真の品質評価を行うバッチ処理クラス"""

    def __init__(self, config_path: Optional[str] = None, logger: Optional[Logger] = None, max_workers: Optional[int] = None, date: Optional[str] = None):
        # max_workersをCPUコア数に応じて動的に設定
        max_workers = max_workers or multiprocessing.cpu_count() // 2  # CPUコア数に基づく設定
        super().__init__(config_path, logger, max_workers)  # 親クラスの初期化
        self._set_directories_and_files(date)  # ディレクトリとファイルの設定
        self.image_data: List[Dict[str, str]] = []  # 画像データを格納するリスト

        # ImageLoaderの初期化（毎回初期化しない）
        self.image_loader = ImageLoader(logger=self.logger)

    def _set_directories_and_files(self, date: Optional[str]) -> None:
        """処理するディレクトリとCSVファイルのパスを設定"""
        self.output_directory = self.config.get('output_directory', 'temp')

        if date:
            self.base_directory = os.path.join(self.config.get('base_directory_root', '/mnt/l/picture/2024'), date)
            self.image_csv_file = os.path.join(self.project_root, self.output_directory, f"{date}_nef_exif_data.csv")
        else:
            self.base_directory = self.config.get('base_directory', '/mnt/l/picture/2024/2024-07-02/')
            self.image_csv_file = os.path.join(self.project_root, self.output_directory,  "nef_exif_data.csv")
        
        self.result_csv_file = os.path.join(self.output_directory, "evaluation_results.csv")

    def load_image_data(self) -> List[Dict[str, str]]:
        """CSVファイルから画像データを読み込む"""
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
        """セットアップフェーズ"""
        self.logger.info("Setting up PortraitQualityBatchProcessor...")

        if not os.path.exists(self.base_directory):
            self.logger.error(f"Image directory not found: {self.base_directory}")
            raise FileNotFoundError(f"Image directory not found: {self.base_directory}")

        if not os.path.exists(self.image_csv_file):
            self.logger.error(f"Input CSV file not found: {self.image_csv_file}")
            raise FileNotFoundError(f"Input CSV file not found: {self.image_csv_file}")

        os.makedirs(self.output_directory, exist_ok=True)
        self.image_data = self.load_image_data()

    def process(self) -> None:
        """メイン処理フェーズで、画像の評価処理を並列実行"""
        self.logger.info("Processing images...")
        with Pool(processes=self.max_workers) as pool:
            results = pool.starmap(self.process_image, [(os.path.join(self.base_directory, img_info["file_name"]),
                                                         img_info["orientation"], img_info["bit_depth"]) 
                                                         for img_info in self.image_data])
        self.save_results(results, self.result_csv_file)

    def cleanup(self) -> None:
        """クリーンアップフェーズ"""
        self.logger.info("Cleaning up resources...")

    def process_image(self, file_name: str, orientation: str, bit_depth: str) -> Dict[str, Optional[Union[str, float, bool]]]:
        """画像の評価を行う処理"""
        self.logger.debug(f"Start processing image: {file_name}")
        try:
            # 画像のロード
            image = self.image_loader.load_image(file_name, output_bps=int(bit_depth), apply_exif_rotation=True, orientation=orientation)
            
            # 顔検出と品質評価
            evaluator = PortraitQualityEvaluator(image)
            evaluation_result = evaluator.evaluate()
            
            return {
                "file_name": file_name,
                "face_detected": evaluation_result.get("face_detected", False),
                "sharpness_evaluation": evaluation_result.get("sharpness_evaluation", None),
                "blurriness_evaluation": evaluation_result.get("blurriness_evaluation", None),
                "contrast_evaluation": evaluation_result.get("contrast_evaluation", None)
            }
        
        except FileNotFoundError as fnfe:
            self.logger.error(f"File not found: {fnfe}")
        except Exception as e:
            self.logger.error(f"Processing failed for {file_name}: {e}", exc_info=True)
        
        return {
            "file_name": file_name, 
            "face_detected": None, 
            "sharpness_evaluation": None, 
            "blurriness_evaluation": None, 
            "contrast_evaluation": None
        }

    def save_results(self, results: List[Dict[str, Optional[Union[str, float, bool]]]], output_file: str) -> None:
        """評価結果をCSVファイルに書き出す"""
        try:
            with open(output_file, 'w', newline='') as file:
                fieldnames = ['file_name', 'face_detected', 'sharpness_evaluation', 'blurriness_evaluation', 'contrast_evaluation']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    writer.writerow(result)
            self.logger.info(f"Results saved to {output_file}.")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

def main():
    """コマンドライン引数を受け取り、バッチ処理を実行するメイン関数"""
    parser = argparse.ArgumentParser(description="Portrait Batch Processor")
    
    # デフォルトの日付を設定
    default_date = "2024-08-26"
    
    parser.add_argument('--date', type=str, default=default_date, help=f"CSVファイルや入力フォルダに対応する日付を指定 (例: 2024-08-26). デフォルトは{default_date}です。")
    parser.add_argument('--max_workers', type=int, help="最大ワーカースレッド数")
    
    args = parser.parse_args()

    # ロガーと設定ファイルのパスを指定してバッチプロセッサを実行
    logger = Logger(logger_name="PortraitQualityBatchProcessor")

    # 現在の実行ディレクトリにconfig_pathを設定
    current_directory = os.getcwd()
    config_path = os.path.join(current_directory, "config.yaml")

    # プロセッサのインスタンスを作成してバッチ処理を実行
    processor = PortraitQualityBatchProcessor(config_path=config_path, logger=logger, date=args.date, max_workers=args.max_workers)
    processor.execute()

if __name__ == "__main__":
    main()
