import argparse
import csv
import datetime
import os
from typing import List, Dict, Optional
import yaml
from concurrent.futures import ThreadPoolExecutor
from batch_framework.base_batch import BaseBatchProcessor

# 定数の定義
FACE_DETECTED = "TRUE"
FACE_SCORE_MULTIPLIER = 2.0  # 顔スコアの倍率
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
    def __init__(self, config_path: str, max_workers: int = 1, max_process_count: int = 5000, date: Optional[str] = None):
        super().__init__(config_path=config_path, max_workers=max_workers, max_process_count=max_process_count)
        self.date = self._parse_date(date)
        self.weights = {}
        self.paths = {}
        self.evaluation_data = []

    def _parse_date(self, date_str: Optional[str]) -> str:
        if date_str:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                self.logger.error("Invalid date format. Using current date.")
        return datetime.datetime.now().strftime("%Y-%m-%d")

    def setup(self) -> None:
        super().setup()
        self.logger.info("Setting up EvaluationRankBatchProcessor.")
        self.load_config(self.config_path)
        
        evaluation_file_path = os.path.join(self.paths.get('evaluation_data_dir', './temp'), f"evaluation_results_{self.date}.csv")
        if not os.path.exists(evaluation_file_path):
            raise FileNotFoundError(f"Evaluation data file not found: {evaluation_file_path}")
        
        self.evaluation_data = self.load_evaluation_data(evaluation_file_path)

    def load_config(self, config_path: str) -> None:
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.weights = config.get('weights', {})
                self.paths = config.get('paths', {})
                self.logger.info("Configuration loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise

    def load_evaluation_data(self, file_path: str) -> List[Dict[str, str]]:
        self.logger.info(f"Loading evaluation data from {file_path}.")
        data = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row['face_detected'] = row.get('face_detected', 'TRUE').upper()
                data.append(row)
        return data

    def calculate_overall_evaluation(self, entry: Dict[str, str]) -> None:
        face_detected = entry.get('face_detected', 'FALSE').upper() == 'TRUE'

        # 顔の評価の加重
        FACE_SHARPNESS_WEIGHT = 0.5
        FACE_CONTRAST_WEIGHT = 0.3
        FACE_NOISE_WEIGHT = 0.1

        # 局所評価の加重（顔部分、全体部分）
        LOCAL_SHARPNESS_WEIGHT = 0.4
        LOCAL_CONTRAST_WEIGHT = 0.3

        # 顔がない場合の評価の加重
        GENERAL_SHARPNESS_WEIGHT = 0.4
        GENERAL_CONTRAST_WEIGHT = 0.3
        GENERAL_NOISE_WEIGHT = 0.2

        # 顔なしの局所評価の加重
        LOCAL_SHARPNESS_WEIGHT_GENERAL = 0.3
        LOCAL_CONTRAST_WEIGHT_GENERAL = 0.2

        if face_detected:
            # 顔の評価（シャープネス、コントラスト、ノイズ）
            face_sharpness = float(entry.get('face_sharpness_score', 0.0) or 0.0)
            face_contrast = float(entry.get('face_contrast_score', 0.0) or 0.0)
            face_noise = float(entry.get('face_noise_score', 0.0) or 0.0)
            
            # 顔部分の局所評価
            local_sharpness = float(entry.get('face_local_sharpness_score', 0.0) or 0.0)
            local_contrast = float(entry.get('face_local_contrast_score', 0.0) or 0.0)
            
            # 顔の評価と局所評価を加重計算
            overall_score = (FACE_SHARPNESS_WEIGHT * face_sharpness +
                            FACE_CONTRAST_WEIGHT * face_contrast -
                            FACE_NOISE_WEIGHT * face_noise +
                            LOCAL_SHARPNESS_WEIGHT * local_sharpness +
                            LOCAL_CONTRAST_WEIGHT * local_contrast)
        else:
            # 顔が検出されない場合の全体評価（シャープネス、コントラスト、ノイズ）
            sharpness = float(entry.get('sharpness_score', 0.0) or 0.0)
            contrast = float(entry.get('contrast_score', 0.0) or 0.0)
            noise = float(entry.get('noise_score', 0.0) or 0.0)
            
            # 全体部分の局所評価
            local_sharpness = float(entry.get('local_sharpness_score', 0.0) or 0.0)
            local_contrast = float(entry.get('local_contrast_score', 0.0) or 0.0)
            
            # 全体の評価と局所評価を加重計算
            overall_score = (GENERAL_SHARPNESS_WEIGHT * sharpness +
                            GENERAL_CONTRAST_WEIGHT * contrast -
                            GENERAL_NOISE_WEIGHT * noise +
                            LOCAL_SHARPNESS_WEIGHT_GENERAL * local_sharpness +
                            LOCAL_CONTRAST_WEIGHT_GENERAL * local_contrast)

        # 計算された総合評価をエントリにセット
        entry['overall_evaluation'] = round(overall_score, 2)

    def get_weights_for_entry(self, entry: Dict[str, str]) -> Dict[str, float]:
        face_detected = entry.get('face_detected', 'FALSE').upper() == 'TRUE'
        return self.weights.get('face_detected' if face_detected else 'no_face_detected', {})

    def assign_acceptance_flag(self, entry: Dict[str, str]) -> None:
        overall_threshold = 75.0
        face_sharpness_threshold = 50.0
        face_contrast_threshold = 30.0
        noise_threshold = 20.0
        face_noise_threshold = 15.0
        contrast_threshold_lower = 15.0  # 最小コントラスト
        contrast_threshold_upper = 25.0  # 最大コントラスト

        overall_score = float(entry.get('overall_evaluation', 0.0) or 0.0)
        face_sharpness = float(entry.get('face_sharpness_score', 0.0) or 0.0)
        face_contrast = float(entry.get('face_contrast_score', 0.0) or 0.0)
        noise = float(entry.get('noise_score', 0.0) or 0.0)
        face_noise = float(entry.get('face_noise_score', 0.0) or 0.0)
        contrast = float(entry.get('contrast_score', 0.0) or 0.0)

        face_detected = entry.get('face_detected', 'FALSE').upper() == 'TRUE'

        # コントラストとシャープネスの統合評価
        if contrast >= contrast_threshold_upper and face_sharpness >= face_sharpness_threshold:
            entry['accepted_flag'] = 1  # 要補正
        elif contrast >= contrast_threshold_lower and face_sharpness >= face_sharpness_threshold:
            entry['accepted_flag'] = 2  # 採用（補正不要）
        else:
            entry['accepted_flag'] = 0  # 不採用

        # 顔が検出されない場合のノイズ評価
        if noise <= noise_threshold:
            entry['accepted_flag'] = 2  # 採用（補正不要）
        else:
            entry['accepted_flag'] = 1  # 要補正

        if overall_score >= overall_threshold:
            if face_detected:
                # 顔のコントラストとシャープネスに基づく統合評価
                if face_contrast >= contrast_threshold_upper and face_sharpness >= face_sharpness_threshold:
                    entry['accepted_flag'] = 1  # 顔のコントラストとシャープネスが高い場合、要補正
                elif face_contrast >= contrast_threshold_lower and face_sharpness >= face_sharpness_threshold:
                    entry['accepted_flag'] = 2  # 顔のコントラストとシャープネスが適切な場合、採用（補正不要）
                else:
                    entry['accepted_flag'] = 0  # 顔のコントラストまたはシャープネスが低すぎる場合、不採用

                # 顔のノイズ評価
                if face_noise <= face_noise_threshold:
                    entry['accepted_flag'] = 2  # 顔のノイズが適切な場合、採用（補正不要）
                else:
                    entry['accepted_flag'] = 1  # 顔のノイズが高すぎる場合、要補正
        else:
            entry['accepted_flag'] = 0  # 不採用

    def process(self) -> None:
        self.logger.info("Processing started.")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self.calculate_overall_evaluation, self.evaluation_data)

        # 採用基準の適用
        for entry in self.evaluation_data:
            self.assign_acceptance_flag(entry)

        face_detected_data = [e for e in self.evaluation_data if e.get('face_detected') == FACE_DETECTED]
        no_face_detected_data = [e for e in self.evaluation_data if e.get('face_detected') != FACE_DETECTED]

        sorted_face_detected = sorted(face_detected_data, key=lambda x: -float(x.get('overall_evaluation', 0.0)))
        sorted_no_face_detected = sorted(no_face_detected_data, key=lambda x: -float(x.get('overall_evaluation', 0.0)))

        def assign_flag(sorted_data):
            threshold_index = max(1, int(len(sorted_data) * 0.35))
            for i, entry in enumerate(sorted_data):
                entry['flag'] = 1 if i < threshold_index else 0

        assign_flag(sorted_face_detected)
        assign_flag(sorted_no_face_detected)

        sorted_data = sorted_face_detected + sorted_no_face_detected

        # 結合後にファイル名順で並べ替え
        sorted_data = sorted(sorted_data, key=lambda x: x.get('file_name', ''))
        
        output_file_path = os.path.join(self.paths.get('output_data_dir', './temp'), f"evaluation_ranking_{self.date}.csv")
        self.output_results(sorted_data, output_file_path)
        self.logger.info("Processing completed.")

    def _process_batch(self, batch=None):
        """ シングル画像処理のため、バッチ処理は不要 """
        self.process()

    def output_results(self, sorted_data: List[Dict[str, str]], output_file_path: str) -> None:
        self.logger.info(f"Outputting results to {output_file_path}.")
        with open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:
            fieldnames = ['file_name', 'face_detected', 'overall_evaluation', 'flag', 'accepted_flag'] + SCORE_TYPES
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in sorted_data:
                writer.writerow({
                    'file_name': entry.get('file_name', 'unknown'),
                    'face_detected': entry.get('face_detected', 'FALSE'),
                    'overall_evaluation': round(entry.get('overall_evaluation', 0.0), 2),
                    'flag': entry.get('flag', 0),
                    'accepted_flag': entry.get('accepted_flag', 0),
                    **{score: round(float(entry.get(score, 0.0) or 0.0), 2) for score in SCORE_TYPES}
                })
        self.logger.info("Results output completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ポートレート評価バッチ処理")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file.")
    parser.add_argument("--date", type=str, help="Date for processing (format: YYYY-MM-DD).")
    args = parser.parse_args()
    processor = EvaluationRankBatchProcessor(config_path=args.config_path, date=args.date)
    processor.execute()
