import argparse
import csv
import datetime
import os
from typing import List, Dict, Optional
import yaml
from batch_framework.base_batch import BaseBatchProcessor

# 定数の定義
FACE_DETECTED = "TRUE"
SCORE_TYPES = [
    'blurriness_score', 
    'sharpness_score', 
    'contrast_score', 
    'noise_score',             # 追加
    'face_contrast_score', 
    'face_sharpness_score',
    'face_noise_score'         # 追加
]

class EvaluationRankBatchProcessor(BaseBatchProcessor):
    """評価結果のランクを計算して出力するバッチ処理クラス"""

    def __init__(self, config_path: str, thresholds_path: str, max_workers: int = 1, max_process_count: int = 5000, date: Optional[str] = None):
        super().__init__(config_path=config_path, max_workers=max_workers, max_process_count=max_process_count)
        self.date = self._parse_date(date)
        self.thresholds = {}  # thresholds.yamlから読み込む閾値
        self.weights = {}
        self.paths = {}
        self.thresholds_path = thresholds_path  # thresholds.yamlのパス
        self.evaluation_data = []

    def _parse_date(self, date_str: Optional[str]) -> str:
        if date_str:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                self.logger.error("Invalid date format. Use 'YYYY-MM-DD'. Using current date as fallback.")
                return datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            return datetime.datetime.now().strftime("%Y-%m-%d")

    def setup(self) -> None:
        """評価結果を読み込むためのセットアップフェーズ"""
        super().setup()  # 親クラスの共通セットアップ処理
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
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise

    def load_thresholds(self, thresholds_path: str) -> None:
        """thresholds.yamlを読み込む"""
        try:
            with open(thresholds_path, mode='r', encoding='utf-8') as file:
                thresholds = yaml.safe_load(file)
                self.thresholds = thresholds
                self.logger.info("Thresholds loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load thresholds: {e}", exc_info=True)
            raise

    def load_evaluation_data(self, file_path: str) -> List[Dict[str, str]]:
        """評価データをロードする"""
        self.logger.info(f"Loading evaluation data from {file_path}.")
        data = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        row['face_detected'] = row.get('face_detected', 'TRUE').upper()
                        data.append(row)
                    except Exception as e:
                        self.logger.warning(f"Skipping invalid row: {row} due to error: {e}")
            self.logger.info("Evaluation data loaded successfully.")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading evaluation data: {e}", exc_info=True)
            raise
        return data

    def calculate_ranks(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """ランクを計算し、並び順を変更"""
        self.logger.info("Calculating ranks using actual scores.")

        for entry in data:
            for score_type in SCORE_TYPES:
                try:
                    score = float(entry.get(score_type, 0))
                except (ValueError, TypeError):
                    score = 0
                
                entry[f'{score_type}_evaluation'] = self.assign_rank(score_type, score)

            self.calculate_overall_evaluation(entry)

        sorted_data = sorted(
            data,
            key=lambda x: (
                -1 if x.get('face_detected') == FACE_DETECTED else 0,
                -x.get('overall_evaluation', 0.0)
            )
        )
        self.logger.info("Ranking completed.")
        return sorted_data

    def assign_rank(self, score_type: str, score: float) -> int:
        """実際のスコアに基づいてランクを割り当てる"""
        levels = self.thresholds.get(score_type, {})
        
        level_1 = float(levels.get('level_1', 0.0))
        level_2 = float(levels.get('level_2', 25.0))
        level_3 = float(levels.get('level_3', 50.0))
        level_4 = float(levels.get('level_4', 75.0))
        level_5 = float(levels.get('level_5', 100.0))

        if score >= level_5:
            return 5
        elif score >= level_4:
            return 4
        elif score >= level_3:
            return 3
        elif score >= level_2:
            return 2
        elif score >= level_1:
            return 1
        else:
            return 0

    def calculate_overall_evaluation(self, entry: Dict[str, str]) -> None:
        """最終評価スコアを計算する"""
        weights = self.get_weights_for_entry(entry)
        overall_score = sum(
            weights.get(score_type, 0.0) * entry.get(f'{score_type}_evaluation', 0)
            for score_type in SCORE_TYPES
        )
        self.logger.info(f"Overall score calculated: {overall_score}")
        entry['overall_evaluation'] = round(overall_score, 2)

    def get_weights_for_entry(self, entry: Dict[str, str]) -> Dict[str, float]:
        """エントリに対する重みを取得する"""
        face_detected = entry.get('face_detected', 'FALSE').upper() == 'TRUE'
        return self.weights.get('face_detected' if face_detected else 'no_face_detected', {})

    def output_results(self, sorted_data: List[Dict[str, str]], output_file_path: str) -> None:
        """最終結果をCSVファイルに出力する"""
        self.logger.info(f"Outputting results to {output_file_path}.")
        with open(output_file_path, mode='w', encoding='utf-8', newline='') as csvfile:
            fieldnames = [
                'file_name',
                'face_detected',
                'face_sharpness_evaluation',
                'face_contrast_evaluation',
                'sharpness_evaluation',
                'contrast_evaluation',
                'blurriness_evaluation',
                'noise_evaluation',          
                'face_noise_evaluation',     
                'overall_evaluation'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in sorted_data:
                writer.writerow({
                    'file_name': entry.get('file_name', 'unknown'),
                    'face_detected': entry.get('face_detected', 'FALSE'),
                    'face_sharpness_evaluation': entry.get('face_sharpness_score_evaluation', 0),
                    'face_contrast_evaluation': entry.get('face_contrast_score_evaluation', 0),
                    'sharpness_evaluation': entry.get('sharpness_score_evaluation', 0),
                    'contrast_evaluation': entry.get('contrast_score_evaluation', 0),
                    'blurriness_evaluation': entry.get('blurriness_score_evaluation', 0),
                    'noise_evaluation': entry.get('noise_score_evaluation', 0),           
                    'face_noise_evaluation': entry.get('face_noise_score_evaluation', 0), 
                    'overall_evaluation': entry.get('overall_evaluation', 0)
                })
        self.logger.info("Results output completed.")

    def process(self) -> None:
        """メイン処理の実装"""
        self.logger.info("Processing started.")
        sorted_data = self.calculate_ranks(self.evaluation_data)
        output_file_path = os.path.join(self.paths.get('output_data_dir', './temp'), f"evaluation_ranking_{self.date}.csv")
        self.output_results(sorted_data, output_file_path)
        self.logger.info("Processing completed.")

    def _process_batch(self, batch: List[Dict[str, str]]) -> None:
        """バッチ単位でランク計算を実行"""
        self.logger.info(f"Processing batch with {len(batch)} entries.")
        sorted_batch = self.calculate_ranks(batch)
        output_file_path = os.path.join(self.paths.get('output_data_dir', './temp'), f"evaluation_ranking_batch_{self.date}.csv")
        self.output_results(sorted_batch, output_file_path)
        self.logger.info("Batch processing completed.")

    def cleanup(self) -> None:
        """後処理の実装"""
        super().cleanup()  # 親クラスの共通クリーンアップ処理
        self.logger.info("Cleanup completed for EvaluationRankBatchProcessor.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ポートレート評価バッチ処理")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file (default: './config/config.yaml').")
    parser.add_argument("--thresholds_path", type=str, help="Path to the thresholds YAML file (default: './config/thresholds.yaml').")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of workers to use.")
    parser.add_argument("--max_process_count", type=int, default=5000, help="Maximum number of processes to execute.")
    parser.add_argument("--date", type=str, help="Date for processing (format: YYYY-MM-DD).")
    args = parser.parse_args()

    processor = EvaluationRankBatchProcessor(
        config_path=args.config_path or "./config/config.yaml",
        thresholds_path=args.thresholds_path or "./config/thresholds.yaml",
        max_workers=args.max_workers,
        max_process_count=args.max_process_count,
        date=args.date
    )
    processor.execute()
