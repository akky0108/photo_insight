import argparse
import csv
import datetime
from typing import List, Dict, Optional
import yaml
from batch_framework.base_batch import BaseBatchProcessor
from log_util import Logger

class EvaluationRankBatchProcessor(BaseBatchProcessor):
    """評価結果のランクを計算して出力するバッチ処理クラス"""

    def __init__(self, config_path: str, max_workers: int, max_process_count: int, logger: Optional[Logger] = None, date: Optional[str] = None):
        super().__init__(config_path=config_path, logger=logger, max_workers=max_workers, max_process_count=max_process_count)
        self.date = self._parse_date(date)
        self.sharpness_thresholds = {}

    def _parse_date(self, date_str: Optional[str]) -> str:
        if date_str:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                self.logger.error("Invalid date format. Use 'YYYY-MM-DD'.")
                raise ValueError("Invalid date format. Use 'YYYY-MM-DD'.")
        else:
            return datetime.datetime.now().strftime("%Y-%m-%d")

    def setup(self) -> None:
        """評価結果を読み込むためのセットアップフェーズ"""
        self.logger.info("Setting up EvaluationRankBatchProcessor.")
        self.load_thresholds_from_config()
        self.evaluation_data = self.load_evaluation_data()

    def run_task(self) -> None:
        """評価データをもとにランクを計算して出力"""
        self.logger.info("Calculating ranks for evaluation results.")
        ranked_results = self.calculate_ranks(self.evaluation_data)
        self.output_ranked_results(ranked_results)

    def cleanup(self) -> None:
        """クリーンアップフェーズ"""
        self.logger.info("Cleaning up EvaluationRankBatchProcessor.")

    def load_thresholds_from_config(self) -> None:
        """YAMLファイルからシャープネスの閾値を読み込む"""
        thresholds_file = "./config/thresholds.yaml"
        try:
            with open(thresholds_file, mode='r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.sharpness_thresholds = config.get('sharpness', {}).get('thresholds', {})
                self.logger.info(f"Loaded sharpness thresholds from config: {self.sharpness_thresholds}")
        except FileNotFoundError:
            self.logger.error(f"Thresholds file not found: {thresholds_file}")
            # デフォルト値を設定
            self.sharpness_thresholds = {'level4': 300, 'level3': 200, 'level2': 150, 'level1': 0}
        except Exception as e:
            self.logger.error(f"Failed to load thresholds: {e}")
            self.sharpness_thresholds = {'level4': 300, 'level3': 200, 'level2': 150, 'level1': 0}

    def load_evaluation_data(self) -> List[Dict[str, str]]:
        file_path = f"./temp/evaluation_results_{self.date}.csv"
        self.logger.info(f"Loading evaluation data from {file_path}.")

        data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
            self.logger.info("Evaluation data loaded successfully.")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        return data

    def calculate_ranks(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        self.logger.info("Calculating ranks with reversed sharpness level evaluation.")

        # 各レベルの閾値を取得
        level4_threshold = self.sharpness_thresholds.get('level4', 300)
        level3_threshold = self.sharpness_thresholds.get('level3', 200)
        level2_threshold = self.sharpness_thresholds.get('level2', 150)

        # シャープネススコアの評価（スコアが高いほどレベルが下がる）
        for entry in data:
            if 'sharpness_score' in entry:
                score = float(entry['sharpness_score'])

                if score >= level4_threshold:
                    entry['sharpness_evaluation'] = "LEVEL4"
                elif score >= level3_threshold:
                    entry['sharpness_evaluation'] = "LEVEL3"
                elif score >= level2_threshold:
                    entry['sharpness_evaluation'] = "LEVEL2"
                else:
                    entry['sharpness_evaluation'] = "LEVEL1"

        sorted_data = sorted(data, key=lambda x: float(x.get('score', 0)), reverse=True)
        for rank, entry in enumerate(sorted_data, start=1):
            entry['rank'] = rank
        return sorted_data

    def output_ranked_results(self, ranked_results: List[Dict[str, str]]) -> None:
        output_file_path = f"./output/ranked_evaluation_results_{self.date}.csv"
        self.logger.info(f"Outputting ranked results to {output_file_path}.")

        fieldnames = ranked_results[0].keys() if ranked_results else []
        try:
            with open(output_file_path, mode='w', encoding='utf-8', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(ranked_results)
            self.logger.info("Ranked results output successfully.")
        except Exception as e:
            self.logger.error(f"Failed to output ranked results: {e}")

def main():
    parser = argparse.ArgumentParser(description="Calculate and output ranks based on evaluation results.")
    parser.add_argument("--date", type=str, help="Target date in 'YYYY-MM-DD' format")
    args = parser.parse_args()

    processor = EvaluationRankBatchProcessor(
        config_path="./config/config.yaml",
        max_workers=4,
        max_process_count=100,
        date=args.date
    )
    processor.execute()

if __name__ == "__main__":
    main()
