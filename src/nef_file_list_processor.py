import os
import csv
import concurrent.futures
import threading
from log_util import Logger
from batch_framework.base_batch import BaseBatchProcessor
from portrait_quality_evaluator import PortraitQualityEvaluator
from image_loader import ImageLoader

class NEFDetailProcessorBatch(BaseBatchProcessor):
    def __init__(self, config_path=None):
        super().__init__(config_path=config_path, logger=Logger())
        self.input_file_path = None
        self.output_file_path = None
        self.temp_file_path = None
        self.processed_details = []
        self.base_directory = '/mnt/l/picture/2024/2024-08-26'
        
        self.evaluation_weights = {
            'sharpness': 0.4,
            'contrast': 0.1,
            'noise': 0.1,
            'wavelet_sharpness': 0.1,
            'face': 0.3
        }
        
        self.processed_rows = []
        self.lock = threading.Lock()

    def setup(self):
        project_root = os.getenv('PROJECT_ROOT')
        if project_root is None:
            self.logger.error("環境変数 'PROJECT_ROOT' が設定されていません。")
            raise EnvironmentError("環境変数 'PROJECT_ROOT' が設定されていません。")
        
        self.input_file_path = os.path.join(project_root, 'temp', 'nef_exif_data.csv')
        self.output_file_path = os.path.join(project_root, 'temp', 'photo_evaluation_results.csv')
        self.temp_file_path = os.path.join(project_root, 'temp', 'temp_results.csv')
        
        if not os.path.exists(self.input_file_path):
            self.logger.error(f"ファイルが見つかりません: {self.input_file_path}")
            raise FileNotFoundError(f"ファイルが見つかりません: {self.input_file_path}")
        
        self.logger.info(f"ファイル '{self.input_file_path}' の処理を開始します。")

    def process_row(self, row):
        nef_file_path = os.path.join(self.base_directory, row['FileName'])
        
        image_loader = ImageLoader(logger=self.logger)
        try:
            rgb_image = image_loader.load_image(nef_file_path)
        except Exception as e:
            row['error'] = f"画像の読み込み中にエラー: {e}"
            self.logger.error(f"ファイル '{nef_file_path}' の読み込みに失敗しました: {e}")
            with self.lock:
                self.processed_rows.append(row)
            return None
        
        evaluator = PortraitQualityEvaluator(rgb_image, weights=self.evaluation_weights)
        evaluation_results = evaluator.evaluate()

        row['overall_score'] = float(evaluation_results['overall_score'])
        row['face_detected'] = bool(evaluation_results.get('face_detected', False))

        # 顔が検出された場合、顔のシャープネス評価を使用
        if row['face_detected']:
            face_evaluation = evaluation_results['face_evaluation']
            blurriness_value = face_evaluation.get('blurriness', 0)
            self.logger.info(f"顔検出あり: {nef_file_path}, 顔のシャープネス値: {blurriness_value}")
        else:
            blurriness_value = evaluation_results.get('blurriness_score', 0)
            self.logger.info(f"顔検出なし: {nef_file_path}, シャープネス値: {blurriness_value}")
        
        # シャープネスの3段階評価
        if blurriness_value > 0.7:
            row['blurriness_level'] = 'High'
        elif blurriness_value > 0.4:
            row['blurriness_level'] = 'Medium'
        else:
            row['blurriness_level'] = 'Low'

        with self.lock:
            self.processed_rows.append(row)
        
        return row

    def process(self):
        try:
            with open(self.input_file_path, 'r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                rows = list(reader)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_row = {executor.submit(self.process_row, row): row for row in rows}
                    for future in concurrent.futures.as_completed(future_to_row):
                        future.result()
            
            self._save_temp_results()

        except Exception as e:
            self.logger.error(f"明細の処理中にエラーが発生しました: {e}")
            raise
        finally:
            self._save_final_results()

    def _save_temp_results(self):
        try:
            with self.lock:
                with open(self.temp_file_path, 'w', newline='', encoding='utf-8') as tempfile:
                    if self.processed_rows:
                        fieldnames = self.processed_rows[0].keys()
                        writer = csv.DictWriter(tempfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in self.processed_rows:
                            writer.writerow(row)
                self.logger.info(f"中間結果が一時ファイル '{self.temp_file_path}' に保存されました。")
        except Exception as e:
            self.logger.error(f"中間結果の保存中にエラーが発生しました: {e}")

    def _save_final_results(self):
        try:
            with open(self.output_file_path, 'w', newline='', encoding='utf-8') as outfile:
                if os.path.exists(self.temp_file_path):
                    with open(self.temp_file_path, 'r', encoding='utf-8') as tempfile:
                        reader = csv.DictReader(tempfile)
                        fieldnames = reader.fieldnames
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in reader:
                            writer.writerow(row)
                    self.logger.info(f"最終結果が '{self.output_file_path}' に保存されました。")
                else:
                    self.logger.warning("一時ファイルが存在しません。")
        except Exception as e:
            self.logger.error(f"最終結果の保存中にエラーが発生しました: {e}")
            raise

    def cleanup(self):
        try:
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)
                self.logger.info("一時ファイルが削除されました。")
        except Exception as e:
            self.logger.error(f"一時ファイルの削除中にエラーが発生しました: {e}")
            raise

        self.logger.info("全ての明細処理が完了しました。")
        self.logger.info(f"{len(self.processed_rows)} 件の明細が処理されました。")

def main():
    process = NEFDetailProcessorBatch()
    process.execute()

if __name__ == "__main__":
    main()
