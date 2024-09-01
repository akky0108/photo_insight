import os
import time
from dotenv import load_dotenv
from file_handler.csv_file_handler import CSVFileHandler
from log_util import Logger  # ログクラスをインポート
from batch_framework.base_batch import BaseBatchProcessor
from composition_evaluator import CompositionEvaluator  # CompositionEvaluator クラスをインポート
import os

class CSVBatchProcess(BaseBatchProcessor):
    def __init__(self, config_path=None, csv_file_path=None ):
        super().__init__(config_path=config_path, logger=Logger())
        self.csv_file_path = csv_file_path
        self.csv_handler = CSVFileHandler()  # 必要に応じて設定を渡す
        self.csv_data = []
        self.output_file_name = None
        self.composition_evaluator = CompositionEvaluator(logger=self.logger)  # 評価クラスのインスタンスを作成

    def setup(self):
        self.logger.info("Setting up resources for CSV processing.")
        if self.csv_file_path:
            try:
                if self.csv_handler.file_exists(self.csv_file_path):
                    self.csv_data = self.csv_handler.read_file(
                        file_path=self.csv_file_path,
                        filters={'face_detected': 'True'},
                        sort_key='overall_score',
                        reverse=True
                    )
                    self.logger.info(f"CSV file loaded successfully: {self.csv_file_path}")
                else:
                    self.logger.error(f"CSV file not found: {self.csv_file_path}")
                    raise FileNotFoundError(f"ファイルが存在しません: {self.csv_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to load CSV file: {e}")
                raise
        else:
            self.logger.error("CSV file path is not provided.")
            raise ValueError("CSV file path is required.")

    def process(self):
        self.logger.info("Processing CSV data.")
        updated_data = []
        for row in self.csv_data:
            updated_row = self.process_row(row)
            updated_data.append(updated_row)
        
        self.csv_data = updated_data
        self.logger.info("CSV data processing complete.")

    def process_row(self, row):
        """明細単位で処理を実行するメソッド"""
        # ファイル名から絶対パスを生成する
        filename = row.get('FileName', '')
        if filename:
            #base_directory = '/mnt/l/picture/2024/2024-07-02'
            base_directory = '/mnt/l/picture/2024/2024-08-26'
            absolute_path = os.path.join(base_directory, filename)
            self.logger.info(f"Converted to absolute path: {absolute_path}")
            
            # 画像をロードし、構図評価を実行する
            try:
                self.composition_evaluator.load_image(absolute_path)
                composition_score = self.composition_evaluator.evaluate_composition()
                row['CompositionScore'] = float(composition_score)
                self.logger.info(f"Evaluated composition score: {composition_score}")
            except Exception as e:
                self.logger.error(f"Failed to evaluate composition for {absolute_path}: {e}")
                row['CompositionScore'] = 'Error'
        
        self.logger.info(f"Processed row: {row}")
        return row

    def cleanup(self):
        self.logger.info("Cleaning up resources after CSV processing.")
        
        # 上位30%を抽出する
        top_30_percent_index = max(1, int(len(self.csv_data) * 0.3))  # 少なくとも1件は出力
        self.csv_data = self.csv_data[:top_30_percent_index]

        self.output_file_name = self.generate_output_file_name()
        try:
            self.csv_handler.write_file(self.output_file_name, self.csv_data)
            self.logger.info(f"Processed CSV data written to: {self.output_file_name}")
        except Exception as e:
            self.logger.error(f"Failed to write processed CSV file: {e}")
            raise

    def generate_output_file_name(self):
        """出力ファイル名を生成するメソッド"""
        base_name = os.path.basename(self.csv_file_path)
        dir_name = os.path.dirname(self.csv_file_path)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(dir_name, f"{name}_processed_top30_{timestamp}{ext}")

if __name__ == "__main__":
    csv_file_path = '/home/mluser/photo_insight/temp/photo_evaluation_results.csv'
    batch_process = CSVBatchProcess(csv_file_path=csv_file_path)
    batch_process.execute()