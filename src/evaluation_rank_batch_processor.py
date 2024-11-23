import argparse
import csv
import datetime
import os
from typing import List, Dict, Optional
import yaml
from batch_framework.base_batch import BaseBatchProcessor
from log_util import Logger

class EvaluationRankBatchProcessor(BaseBatchProcessor):
    """評価結果のランクを計算して出力するバッチ処理クラス"""

    def __init__(self, config_path: str, thresholds_path: str, max_workers: int, max_process_count: int, logger: Optional[Logger] = None, date: Optional[str] = None):
        super().__init__(config_path=config_path, logger=logger, max_workers=max_workers, max_process_count=max_process_count)
        self.date = self._parse_date(date)
        self.thresholds = {}  # thresholds.yamlから読み込む閾値
        self.weights = {}
        self.paths = {}
        self.thresholds_path = thresholds_path  # thresholds.yamlのパス

    def _parse_date(self, date_str: Optional[str]) -> str:
        if date_str:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                self.logger.error("Invalid date format. Use 'YYYY-MM-DD'.")
                raise ValueError("Invalid date format. Use 'YYYY-MM-DD'.")
        else:
            return datetime.datetime.now().strftime("%Y-%m-%d")

    def setup(self) -> None:
        """評価結果を読み込むためのセットアップフェーズ"""
        self.logger.info("Setting up EvaluationRankBatchProcessor.")
        
        self.load_config(self.config_path)  # config.yamlの読み込み
        self.load_thresholds(self.thresholds_path)  # thresholds.yamlの読み込み
        
        evaluation_file_path = os.path.join(self.paths.get('evaluation_data_dir', './temp'), f"evaluation_results_{self.date}.csv")
        
        if not os.path.exists(evaluation_file_path):
            self.logger.error(f"Evaluation data file not found: {evaluation_file_path}")
            raise FileNotFoundError(f"Evaluation data file not found: {evaluation_file_path}")

        self.evaluation_data = self.load_evaluation_data(evaluation_file_path)

    def load_config(self, config_path: str) -> None:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, mode='r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.weights = config.get('weights', {})
                self.paths = config.get('paths', {})
                self.logger.info("Configuration loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def load_thresholds(self, thresholds_path: str) -> None:
        """thresholds.yamlを読み込む"""
        try:
            with open(thresholds_path, mode='r', encoding='utf-8') as file:
                thresholds = yaml.safe_load(file)
                self.thresholds = thresholds
                self.logger.info("Thresholds loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load thresholds: {e}")
            raise

    def load_evaluation_data(self, file_path: str) -> List[Dict[str, str]]:
        """評価データをロードする"""
        self.logger.info(f"Loading evaluation data from {file_path}.")
        
        data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    row['face_detected'] = row.get('face_detected', 'TRUE').upper()
                    data.append(row)
            self.logger.info("Evaluation data loaded successfully.")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading evaluation data: {e}")
            raise
        return data

    def calculate_ranks(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """ランクを計算する（データセットの最大値・最小値で正規化）"""
        self.logger.info("Calculating ranks with dataset-based normalization.")
        
        # 各スコアタイプの最小値と最大値を計算
        score_types = ['blurriness_score', 'contrast_score', 'face_contrast_score', 'face_sharpness_score', 'sharpness_score']

        # スコアの最小値と最大値を初期化
        score_min_max = {score_type: {"min": float('inf'), "max": float('-inf')} for score_type in score_types}
        
        # データセット内の最大値と最小値を計算
        for entry in data:
            for score_type in score_types:
                try:
                    raw_score = float(entry.get(score_type, 0) or 0)
                    if raw_score < score_min_max[score_type]["min"]:
                        score_min_max[score_type]["min"] = raw_score
                    if raw_score > score_min_max[score_type]["max"]:
                        score_min_max[score_type]["max"] = raw_score
                except ValueError:
                    continue

        # 正規化の計算
        def normalize_score(score, score_type):
            min_val = score_min_max[score_type]["min"]
            max_val = score_min_max[score_type]["max"]
            return (score - min_val) / (max_val - min_val) if max_val > min_val else 0

        for entry in data:
            try:
                # 各スコアを取得し、データセットの最小値・最大値で正規化
                for score_type in score_types:
                    raw_score = float(entry.get(score_type, 0) or 0)
                    normalized_score = normalize_score(raw_score, score_type)
                    entry[f'{score_type}_normalized'] = normalized_score

                    # 各スコアタイプごとにランクを設定
                    entry[f'{score_type}_evaluation'] = self.assign_rank(score_type, normalized_score)

                # 顔が検出された場合、顔に関連するスコアを評価
                if entry.get('face_detected') == 'TRUE':
                    entry['face_sharpness_evaluation'] = self.assign_rank('face_sharpness_score', entry.get('face_sharpness_score', 0))
                    entry['face_contrast_evaluation'] = self.assign_rank('face_contrast_score', entry.get('face_contrast_score', 0))

                # スコアを重み付けして、最終的な評価を計算
                self.calculate_overall_evaluation(entry)

            except (ValueError, TypeError) as e:
                self.logger.warning(f"Invalid data encountered: {entry}, Error: {e}")
                entry['overall_evaluation'] = 0.0

        sorted_data = sorted(data, key=lambda x: -x.get('overall_evaluation', 0.0))
        self.logger.info("Ranking completed.")
        return sorted_data

    def assign_rank(self, score_type: str, normalized_score: float) -> int:
        """スコアタイプごとにランクを割り当てる"""
        levels = self.thresholds.get(score_type, {})
        level_1 = levels.get('level_1', 0.0)
        level_2 = levels.get('level_2', 0.25)
        level_3 = levels.get('level_3', 0.5)
        level_4 = levels.get('level_4', 0.75)
        level_5 = levels.get('level_5', 1.0)

        if normalized_score >= level_5:
            return 5
        elif normalized_score >= level_4:
            return 4
        elif normalized_score >= level_3:
            return 3
        elif normalized_score >= level_2:
            return 2
        elif normalized_score >= level_1:
            return 1
        else:
            return 0

    def calculate_overall_evaluation(self, entry: Dict[str, str]) -> None:
        """最終評価スコアを計算する"""
        weights = self.weights.get('face_detected' if entry.get('face_detected', 'FALSE') == 'TRUE' else 'no_face_detected', {})
        weight_face_sharpness = weights.get('face_sharpness', 0.0)
        weight_face_contrast = weights.get('face_contrast', 0.0)
        weight_sharpness = weights.get('overall_sharpness', 0.0)
        weight_contrast = weights.get('overall_contrast', 0.0)

        overall_score = (
            weight_face_sharpness * entry.get('face_sharpness_evaluation', 0) +
            weight_face_contrast * entry.get('face_contrast_evaluation', 0) +
            weight_sharpness * entry.get('sharpness_evaluation', 0) +
            weight_contrast * entry.get('contrast_evaluation', 0)
        )
        entry['overall_evaluation'] = round(overall_score, 3)

    def output_results(self, sorted_data: List[Dict[str, str]]) -> None:
        """評価結果を出力する"""
        output_file_path = os.path.join(self.paths.get('output_data_dir', './temp'), f"evaluation_ranking_{self.date}.csv")
        self.logger.info(f"Outputting results to {output_file_path}.")
        
        fieldnames = [
            'file_name',
            'sharpness_evaluation',
            'face_sharpness_evaluation',
            'contrast_evaluation',
            'face_contrast_evaluation',
            'overall_evaluation'
        ]
        
        try:
            with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for entry in sorted_data:
                    # デバッグログ：出力内容を確認
                    self.logger.debug(f"Writing entry to CSV: {entry}")

                    writer.writerow({
                        'file_name': entry['file_name'],
                        'sharpness_evaluation': entry.get('sharpness_evaluation', 0),
                        'face_sharpness_evaluation': entry.get('face_sharpness_evaluation', 0),
                        'contrast_evaluation': entry.get('contrast_evaluation', 0),
                        'face_contrast_evaluation': entry.get('face_contrast_evaluation', 0),
                        'overall_evaluation': entry['overall_evaluation']
                    })
            self.logger.info(f"Results successfully output to {output_file_path}.")
        except Exception as e:
            self.logger.error(f"Failed to write results: {e}")
            raise

    def process(self) -> None:
        """プロセスを開始する"""
        self.logger.info("Starting the evaluation ranking process.")
        sorted_data = self.calculate_ranks(self.evaluation_data)
        self.output_results(sorted_data)

    def cleanup(self) -> None:
        """クリーンアップ処理"""
        self.logger.info("Cleaning up EvaluationRankBatchProcessor resources...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ポートレート評価バッチ処理")
    parser.add_argument("--config_path", type=str, default="/home/mluser/photo_insight/config/config.yaml", help="Config file path") 
    parser.add_argument("--thresholds_path", type=str, default="/home/mluser/photo_insight/config/thresholds.yaml", help="Thresholds file path") 
    parser.add_argument("--max_workers", type=int, default=1, help="Number of max workers for processing")
    parser.add_argument("--max_process_count", type=int, default=1, help="Max process count")
    parser.add_argument("--date", type=str, help="Evaluation date (YYYY-MM-DD)")

    args = parser.parse_args()

    logger = Logger("evaluation_rank_batch")
    processor = EvaluationRankBatchProcessor(
        config_path=args.config_path,
        thresholds_path=args.thresholds_path,
        max_workers=args.max_workers,
        max_process_count=args.max_process_count,
        logger=logger,
        date=args.date
    )
    processor.setup()
    processor.process()
    processor.cleanup()
