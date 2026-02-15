import os
import time
from file_handler.csv_file_handler import CSVFileHandler
from photo_insight.utils.app_logger import Logger  # ログクラスをインポート
from photo_insight.batch_framework.base_batch import BaseBatchProcessor
from composition_evaluator import (
    CompositionEvaluator,
)  # CompositionEvaluator クラスをインポート


class CSVBatchProcess(BaseBatchProcessor):
    def __init__(self, config_path=None, csv_file_path=None):
        super().__init__(config_path=config_path, logger=Logger())
        self.csv_file_path = csv_file_path
        self.csv_handler = CSVFileHandler()  # 必要に応じて設定を渡す
        self.csv_data = []
        self.output_file_name = None
        self.composition_evaluator = CompositionEvaluator(
            logger=self.logger
        )  # 評価クラスのインスタンスを作成

    def setup(self):
        self.logger.info("CSV処理のリソースをセットアップしています。")
        if self.csv_file_path:
            try:
                if self.csv_handler.file_exists(self.csv_file_path):
                    self.csv_data = self.csv_handler.read_file(
                        file_path=self.csv_file_path,
                        filters={"face_detected": "True"},
                        sort_key="overall_score",
                        reverse=True,
                    )
                    self.logger.info(
                        f"CSVファイルの読み込みに成功しました: {self.csv_file_path}"
                    )
                else:
                    self.logger.error(
                        f"CSVファイルが見つかりません: {self.csv_file_path}"
                    )
                    raise FileNotFoundError(
                        f"ファイルが存在しません: {self.csv_file_path}"
                    )
            except Exception as e:
                self.logger.error(f"CSVファイルの読み込みに失敗しました: {e}")
                raise
        else:
            self.logger.error("CSVファイルパスが指定されていません。")
            raise ValueError("CSVファイルパスは必須です。")

    def process(self):
        self.logger.info("CSVデータを処理しています。")
        updated_data = []
        for row in self.csv_data:
            updated_row = self.process_row(row)
            updated_data.append(updated_row)

        self.csv_data = updated_data
        self.logger.info("CSVデータの処理が完了しました。")

    def process_row(self, row):
        """明細単位で処理を実行するメソッド"""
        filename = row.get("FileName", "")
        if filename:
            base_directory = "/mnt/l/picture/2024/2024-08-26"
            absolute_path = os.path.join(base_directory, filename)
            self.logger.info(f"絶対パスに変換されました: {absolute_path}")

            try:
                self.composition_evaluator.load_image(absolute_path)
                composition_score = self.composition_evaluator.evaluate_composition()
                row["CompositionScore"] = float(composition_score)
                self.logger.info(f"構図スコアが評価されました: {composition_score}")
            except Exception as e:
                self.logger.error(f"{absolute_path} の構図評価に失敗しました: {e}")
                row["CompositionScore"] = "Error"

        self.logger.info(f"処理された行: {row}")
        return row

    def cleanup(self):
        self.logger.info("CSV処理後のリソースをクリーンアップしています。")

        # 上位30%を抽出する
        top_30_percent_index = max(
            1, int(len(self.csv_data) * 0.3)
        )  # 少なくとも1件は出力
        self.csv_data = self.csv_data[:top_30_percent_index]

        self.output_file_name = self.generate_output_file_name()
        try:
            self.csv_handler.write_file(self.output_file_name, self.csv_data)
            self.logger.info(
                f"処理されたCSVデータが出力されました: {self.output_file_name}"
            )
        except Exception as e:
            self.logger.error(f"処理されたCSVファイルの書き込みに失敗しました: {e}")
            raise

    def generate_output_file_name(self):
        """出力ファイル名を生成するメソッド"""
        base_name = os.path.basename(self.csv_file_path)
        dir_name = os.path.dirname(self.csv_file_path)
        name, ext = os.path.splitext(base_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(dir_name, f"{name}_processed_top30_{timestamp}{ext}")


if __name__ == "__main__":
    csv_file_path = "/home/mluser/photo_insight/temp/photo_evaluation_results.csv"
    batch_process = CSVBatchProcess(csv_file_path=csv_file_path)
    batch_process.execute()
