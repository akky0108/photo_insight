import os
import csv
import threading
import numpy as np
from typing import List, Optional
import argparse
from batch_framework.base_batch import BaseBatchProcessor
from image_loader import ImageLoader
from portrait_quality_evaluator import PortraitQualityEvaluator

# スレッドセーフなロックオブジェクト
lock = threading.Lock()

class PortraitBatchProcessor(BaseBatchProcessor):
    def __init__(self, config_path: Optional[str] = None, max_workers: int = 2, date: Optional[str] = None):
        """
        PortraitBatchProcessor の初期化。設定ファイルと画像ファイルのリストを初期化する。
        
        :param config_path: 設定ファイルのパス（省略可能）
        :param max_workers: 並列で実行するワーカーの最大数
        :param date: 対象とする日付（フォーマット例: '2024-08-26'）
        """
        super().__init__(config_path)
        self.image_files: List[str] = []
        
        # ディレクトリの設定（指定された日付をフォルダ名に使用）
        if date:
            self.base_directory = os.path.join(self.config.get('base_directory_root', '/mnt/l/picture/2024'), date)
            self.image_csv_file = f"{date}_nef_exif_data.csv"
        else:
            self.base_directory = self.config.get('base_directory', '/mnt/l/picture/2024/2024-08-26/')
            self.image_csv_file = "nef_exif_data.csv"
        
        self.output_directory = self.config.get('output_directory', 'temp')

        # ImageLoader の初期化
        self.loader = ImageLoader(self.logger)
        
        # 結果CSVファイルの初期化
        self.result_csv_file = os.path.join(self.output_directory, "evaluation_results.csv")

    def setup(self) -> None:
        """
        バッチ処理の初期設定を行う。画像ファイルのリストをロードし、結果CSVファイルを準備する。
        """
        self.logger.info("Portrait batch processor setup started.")
        
        # 画像ファイルのリストを指定された日付のCSVからロード
        image_csv_path = os.path.join(self.project_root, self.output_directory, self.image_csv_file)
        self.image_files = self._load_image_files_from_csv(image_csv_path)
        self.logger.info(f"Found {len(self.image_files)} image files listed in {image_csv_path}.")

        # 結果CSVファイルの作成
        self._initialize_csv_files()

    def process(self) -> None:
        """
        画像の評価処理を行う。各画像ファイルについて、評価を実施し結果をCSVファイルに記録する。
        """
        self.logger.info("Processing images for portrait evaluation.")
        
        # 各画像ファイルに対して評価を実施
        for image_file in self.image_files:
            try:
                image_path = os.path.join(self.base_directory, image_file)

                # 画像をスレッドセーフにロード
                with lock:
                    image = self._load_image(image_path)

                if image is not None:
                    evaluator = PortraitQualityEvaluator(image_path_or_array=image, is_raw=False, logger=self.logger)
                    evaluation_result = evaluator.evaluate()

                    # 評価結果をCSVファイルに追記
                    self._append_evaluation_result(image_file, evaluation_result)
                else:
                    self.logger.warning(f"Failed to load image: {image_file}")
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")

    def cleanup(self) -> None:
        """
        バッチ処理後のクリーンアップ処理を行う。主にログ出力を行う。
        """
        self.logger.info("Cleaning up after batch process.")

    def _load_image_files_from_csv(self, csv_path: str) -> List[str]:
        """
        指定されたCSVファイルから画像ファイルのリストをロードする。
        
        :param csv_path: 画像ファイル名が記載されたCSVファイルのパス
        :return: ロードされた画像ファイル名のリスト
        """
        image_files = []
        try:
            with open(csv_path, mode='r') as csvfile:
                reader = csv.reader(csvfile)
                image_files = [row[0] for row in reader if row]
        except Exception as e:
            self.logger.error(f"Error reading CSV file {csv_path}: {e}")
        return image_files

    def _load_image(self, file_path: str) -> np.ndarray:
        """
        ImageLoader クラスを使用して、画像をロードする。
        
        :param file_path: ロードする画像ファイルのパス
        :return: ロードされた画像のnumpy配列
        """
        try:
            return self.loader.load_image(file_path, output_bps=8)
        except Exception as e:
            self.logger.error(f"Error loading image {file_path}: {e}")
            return None

    def _initialize_csv_files(self) -> None:
        """
        結果CSVファイルを初期化する。ヘッダー行を追加する。
        """
        with open(self.result_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Image File", "Overall Score", "Sharpness", "Contrast", "Noise", 
                "Wavelet Sharpness", "Blurriness", "Face Detected", "Face Evaluation"
            ])

    def _append_evaluation_result(self, image_file: str, result: dict) -> None:
        """
        評価結果を結果CSVファイルに追記する。
        
        :param image_file: 評価対象の画像ファイル名
        :param result: 画像評価結果の辞書
        """
        # 各評価指標の結果を取得。キーが存在しない場合は 'N/A' をデフォルトとして使用
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

# コマンドライン引数で日付を指定
def main():
    """
    バッチプロセスの実行エントリーポイント
    """
    parser = argparse.ArgumentParser(description="Portrait Batch Processor")
    parser.add_argument('--date', type=str, help="Specify the date (e.g., 2024-08-26) for the corresponding CSV file and input folder.")
    args = parser.parse_args()

    # 指定された日付を使ってプロセッサを初期化
    process = PortraitBatchProcessor(date=args.date)
    process.execute()

if __name__ == "__main__":
    main()
