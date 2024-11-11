import os
import csv
import yaml
from typing import List, Dict
from batch_framework.base_batch import BaseBatchProcessor
from utils.math_utils import MathUtils  # MathUtilsクラスをインポート
import numpy as np

class ThresholdCalculatorBatchProcessor(BaseBatchProcessor):
    """
    シャープネスやブレなどの評価指標の閾値を計算するバッチ処理クラス。
    """

    def __init__(self, config_path: str, max_workers: int, max_process_count: int):
        super().__init__(config_path=config_path, max_workers=max_workers, max_process_count=max_process_count)
        self.thresholds = {}
        self.keywords = "evaluation_results"

    def setup(self) -> None:
        """
        データの読み込みとセットアップを行う。
        """
        self.logger.info("Setting up ThresholdCalculatorBatchProcessor.")
        self.evaluation_data = self.load_evaluation_data()
        self.logger.info(f"Loaded {len(self.evaluation_data)} evaluation records.")

    def process(self) -> None:
        """
        閾値を計算し、結果を保存する。
        """
        self.logger.info("Calculating thresholds.")
        self.calculate_thresholds()
        self.save_thresholds()

    def cleanup(self) -> None:
        """
        クリーンアップ処理。
        """
        self.logger.info("Cleaning up ThresholdCalculatorBatchProcessor.")

    def load_evaluation_data(self) -> List[Dict[str, str]]:
        """
        tempフォルダから評価データを読み込む。

        :return: 読み込んだデータのリスト
        """
        temp_folder = "./temp"
        data = []

        # tempフォルダ内のファイルをチェックして、キーワードが含まれるファイルを読み込む
        for filename in os.listdir(temp_folder):
            if self.keywords in filename and filename.endswith(".csv"):
                file_path = os.path.join(temp_folder, filename)
                self.logger.info(f"Loading evaluation data from {file_path}.")

                try:
                    with open(file_path, mode='r', encoding='utf-8') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            # スコアが空白でない場合のみデータを追加
                            if (
                                row.get('sharpness_score') and 
                                row.get('blurriness_score') and 
                                row.get('face_sharpness_score')
                            ):
                                data.append(row)
                except FileNotFoundError:
                    self.logger.error(f"File not found: {file_path}")
                    continue
        return data

    def calculate_thresholds(self) -> None:
        """
        各評価指標の閾値を計算する。
        """
        # 空白や無効なスコアを除外してデータを収集
        sharpness_scores = [
            float(row['sharpness_score']) for row in self.evaluation_data 
            if 'sharpness_score' in row and row['sharpness_score'].strip()
        ]
        blurriness_scores = [
            float(row['blurriness_score']) for row in self.evaluation_data 
            if 'blurriness_score' in row and row['blurriness_score'].strip()
        ]
        face_sharpness_scores = [
            float(row['face_sharpness_score']) for row in self.evaluation_data 
            if 'face_sharpness_score' in row and row['face_sharpness_score'].strip()
        ]

        # 閾値の計算
        self.thresholds['sharpness_score'] = self.calculate_percentile_thresholds(sharpness_scores)
        self.thresholds['blurriness_score'] = self.calculate_percentile_thresholds(blurriness_scores)
        self.thresholds['face_sharpness_score'] = self.calculate_percentile_thresholds(face_sharpness_scores)

    def calculate_percentile_thresholds(self, scores: List[float]) -> Dict[str, float]:
        """
        スコアのリストから4段階の閾値を計算する。

        :param scores: スコアのリスト
        :return: 閾値の辞書
        """
        if not scores:
            return {'level_1': 0, 'level_2': 0, 'level_3': 0, 'level_4': 0}
        
        scores = np.array(scores)
        thresholds = {
            'level_1': float(np.percentile(scores, 25)),  # 下位25%
            'level_2': float(np.percentile(scores, 50)),  # 中央値
            'level_3': float(np.percentile(scores, 75)),  # 上位25%
            'level_4': float(np.percentile(scores, 100)), # 最大値
        }
        return thresholds

    def save_thresholds(self) -> None:
        """
        計算した閾値をconfigフォルダに保存する。
        """
        output_path = "./config/thresholds.yaml"
        self.logger.info(f"Saving thresholds to {output_path}.")

        try:
            with open(output_path, mode='w', encoding='utf-8') as yamlfile:
                yaml.dump(self.thresholds, yamlfile)
            self.logger.info("Thresholds saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save thresholds: {e}")

# エントリポイント
if __name__ == "__main__":
    processor = ThresholdCalculatorBatchProcessor(
        config_path="./config/config.yaml",
        max_workers=4,
        max_process_count=100
    )
    processor.execute()
