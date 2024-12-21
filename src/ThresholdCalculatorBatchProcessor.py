import os
import pandas as pd
import yaml
import numpy as np
from typing import List, Dict
from batch_framework.base_batch import BaseBatchProcessor

class ThresholdCalculator(BaseBatchProcessor):
    def __init__(self, config_path: str, input_folder: str, output_path: str):
        super().__init__(config_path=config_path)
        self.input_folder = input_folder
        self.output_path = output_path
        self.evaluation_data = pd.DataFrame()
        self.thresholds = {}

    def setup(self) -> None:
        super().setup()  # 親クラスの共通セットアップ処理
        self.logger.info("Setting up ThresholdCalculator.")
        self.evaluation_data = self.load_evaluation_data()
        self.logger.info(f"Loaded {len(self.evaluation_data)} evaluation records.")

    def process(self) -> None:
        self.logger.info("Calculating thresholds.")
        self._process_batch(self.evaluation_data)
        self.save_thresholds()

    def cleanup(self) -> None:
        super().cleanup()  # 親クラスの共通クリーンアップ処理
        self.logger.info("Cleanup completed for ThresholdCalculator.")

    def load_evaluation_data(self) -> pd.DataFrame:
        all_data = []
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".csv") and "evaluation_results_" in filename:
                file_path = os.path.join(self.input_folder, filename)
                self.logger.info(f"Loading evaluation data from {file_path}.")
                try:
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                except Exception as e:
                    self.logger.error(f"Error reading {file_path}: {e}")

        if all_data:
            data = pd.concat(all_data, ignore_index=True)
            return data
        else:
            self.logger.warning("No evaluation data found.")
            return pd.DataFrame()

    def _process_batch(self, batch_data: pd.DataFrame) -> None:
        # 新しいスコアを含めた評価指標リスト
        score_columns = [
            'sharpness_score', 
            'blurriness_score', 
            'contrast_score', 
            'face_sharpness_score', 
            'face_contrast_score',
            'face_noise_score',      # 追加指標
            'noise_score'             # 追加指標
        ]
        
        for score_column in score_columns:
            scores = self.extract_scores(score_column, batch_data)
            if scores:
                self.thresholds[score_column] = self.calculate_thresholds_for_scores(scores)
            else:
                self.thresholds[score_column] = self.get_zero_thresholds()

    def extract_scores(self, key: str, batch_data: pd.DataFrame) -> List[float]:
        if key in batch_data.columns:
            valid_data = batch_data[key].dropna()
            return [self.convert_to_float(value) for value in valid_data]
        return []

    def convert_to_float(self, value) -> float:
        if isinstance(value, np.generic):
            return float(value)
        return float(value)

    def calculate_thresholds_for_scores(self, scores: List[float]) -> Dict[str, float]:
        # 四分位数を計算
        q1 = np.percentile(scores, 25)
        q2 = np.median(scores)
        q3 = np.percentile(scores, 75)
        
        # 最小値と最大値を取得
        min_score = min(scores)
        max_score = max(scores)
        
        # 閾値の設定
        thresholds = {
            'level_1': min_score,
            'level_2': q1,
            'level_3': q2,
            'level_4': q3,
            'level_5': max_score
        }

        return thresholds

    def get_zero_thresholds(self) -> Dict[str, float]:
        return {
            'level_1': 0.0,
            'level_2': 0.0,
            'level_3': 0.0,
            'level_4': 0.0,
            'level_5': 0.0
        }

    def save_thresholds(self) -> None:
        self.logger.info(f"Saving thresholds to {self.output_path}.")
        try:
            thresholds_for_yaml = {key: {k: float(v) for k, v in value.items()} for key, value in self.thresholds.items()}
            with open(self.output_path, mode='w', encoding='utf-8') as yamlfile:
                yaml.dump(thresholds_for_yaml, yamlfile, default_flow_style=False, allow_unicode=True)
            self.logger.info("Thresholds saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save thresholds: {e}")

# メイン処理
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate thresholds for evaluation data.")
    parser.add_argument("--config_path", type=str, default="./config/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--input_folder", type=str, default="./temp/", help="Path to the folder containing evaluation data.")
    parser.add_argument("--output_path", type=str, default="./config/thresholds.yaml", help="Path to save the thresholds YAML file.")
    args = parser.parse_args()

    processor = ThresholdCalculator(
        config_path=args.config_path,
        input_folder=args.input_folder,
        output_path=args.output_path
    )
    processor.execute()
