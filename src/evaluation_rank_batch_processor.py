import argparse
import csv
import datetime
import os
import uuid
from typing import List, Dict, Optional
import yaml
from collections import defaultdict
from threading import Lock
from batch_framework.base_batch import BaseBatchProcessor

FACE_DETECTED = "TRUE"

FACE_WEIGHTS = {
    "sharpness": 0.5,
    "contrast": 0.3,
    "noise": 0.1,
    "local_sharpness": 0.4,
    "local_contrast": 0.3,
}

GENERAL_WEIGHTS = {
    "sharpness": 0.4,
    "contrast": 0.3,
    "noise": 0.2,
    "local_sharpness": 0.3,
    "local_contrast": 0.2,
}

EXTRA_WEIGHTS = {
    "composition": 0.1,
    "position": 0.05,
    "framing": 0.05,
    "direction": 0.05,
}

SCORE_TYPES: List[str] = [
    "blurriness_score", "sharpness_score", "contrast_score", "noise_score",
    "local_sharpness_score", "local_contrast_score",
    "face_sharpness_score", "face_contrast_score", "face_noise_score",
    "face_local_sharpness_score", "face_local_contrast_score",
    "composition_rule_based_score", "face_position_score",
    "framing_score", "face_direction_score", "eye_contact_score",
]

def weighted_sum(values: Dict[str, float], weights: Dict[str, float]) -> float:
    """指定された値と重みの積和を返す"""
    return sum(values.get(k, 0.0) * weights.get(k, 0.0) for k in weights)


class EvaluationRankBatchProcessor(BaseBatchProcessor):
    def __init__(
        self,
        config_path: str,
        max_workers: int = 1,
        date: Optional[str] = None,
    ) -> None:
        super().__init__(config_path=config_path, max_workers=max_workers)
        self.date = self._parse_date(date)
        self.weights: Dict[str, Dict[str, float]] = {}
        self.paths: Dict[str, str] = {}

    def _parse_date(self, date_str: Optional[str]) -> str:
        """文字列から日付をパース。無効なら今日の日付を返す"""
        if date_str:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                self.logger.error("Invalid date format. Using current date.")
        return datetime.datetime.now().strftime("%Y-%m-%d")

    def setup(self) -> None:
        """設定ファイルの読み込みと評価CSVファイルのパス構築"""
        super().setup()
        self.logger.info("Setting up EvaluationRankBatchProcessor.")
        self.load_config(self.config_path)

        self.output_data = []  # 評価データのメモリ保持
        self._data_lock = Lock()  # 排他制御

        self.evaluation_csv_path = os.path.join(
            self.paths.get("evaluation_data_dir", "./temp"),
            f"evaluation_results_{self.date}.csv",
        )

        if not os.path.exists(self.evaluation_csv_path):
            raise FileNotFoundError(f"Evaluation data file not found: {self.evaluation_csv_path}")

    def load_config(self, config_path: str) -> None:
        """YAML設定ファイルからパスと重みを読み込む"""
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                self.weights = config.get("weights", {})
                self.paths = config.get("paths", {})
                self.logger.info("Configuration loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}", exc_info=True)
            raise

    def load_evaluation_data(self, file_path: str) -> List[Dict[str, str]]:
        """CSVファイルから評価データを読み込む"""
        self.logger.info(f"Loading evaluation data from {file_path}.")
        with open(file_path, "r", encoding="utf-8") as csvfile:
            return [
                {**row, "face_detected": row.get("face_detected", "TRUE").upper()}
                for row in csv.DictReader(csvfile)
            ]

    def get_data(self) -> List[Dict[str, str]]:
        """BaseBatchProcessor に準拠して評価データを取得"""
        return self.load_evaluation_data(self.evaluation_csv_path)

    def process(self) -> None:
        """BaseBatchProcessor に処理フェーズを委譲"""
        self.logger.info("Processing started.")
        super().process()
        self.logger.info("Processing completed.")

    def _process_batch(self, batch: List[Dict[str, str]]) -> None:
        """1バッチ分の処理（評価、フラグ、ランク付け、出力）を行う"""
        self.evaluate_batch_entries(batch)
        self.assign_flags_to_entries(batch)
        self.rank_and_flag_top_entries(batch)

        with self._data_lock:
            self.output_data.extend(batch)

    def evaluate_batch_entries(self, batch: List[Dict[str, str]]) -> None:
        """各エントリのスコアを集計して overall_evaluation を付与"""
        for entry in batch:
            self.calculate_overall_evaluation(entry)

    def calculate_overall_evaluation(self, entry: Dict[str, str]) -> None:
        """個々のエントリに対してスコアの重み付き合計を計算"""
        face_detected = entry.get("face_detected", "FALSE") == "TRUE"

        if face_detected:
            values = {
                "sharpness": float(entry.get("face_sharpness_score", 0.0)),
                "contrast": float(entry.get("face_contrast_score", 0.0)),
                "noise": float(entry.get("face_noise_score", 0.0)),
                "local_sharpness": float(entry.get("face_local_sharpness_score", 0.0)),
                "local_contrast": float(entry.get("face_local_contrast_score", 0.0)),
            }
            weights = self.weights.get("face", FACE_WEIGHTS)
        else:
            values = {
                "sharpness": float(entry.get("sharpness_score", 0.0)),
                "contrast": float(entry.get("contrast_score", 0.0)),
                "noise": float(entry.get("noise_score", 0.0)),
                "local_sharpness": float(entry.get("local_sharpness_score", 0.0)),
                "local_contrast": float(entry.get("local_contrast_score", 0.0)),
            }
            weights = self.weights.get("general", GENERAL_WEIGHTS)

        base_score = weighted_sum(values, weights)

        extra_values = {
            "composition": float(entry.get("composition_rule_based_score", 0.0)),
            "position": float(entry.get("face_position_score", 0.0)),
            "framing": float(entry.get("framing_score", 0.0)),
            "direction": float(entry.get("face_direction_score", 0.0)),
        }
        extra_weights = self.weights.get("extra", EXTRA_WEIGHTS)
        extra_score = weighted_sum(extra_values, extra_weights)

        entry["overall_evaluation"] = round(base_score + extra_score, 2)

    def assign_flags_to_entries(self, batch: List[Dict[str, str]]) -> None:
        """各エントリに accepted_flag を割り当て"""
        for entry in batch:
            self.assign_acceptance_flag(entry)

    def assign_acceptance_flag(self, entry: Dict[str, str]) -> None:
        """accepted_flag をスコアや閾値に基づいて設定する"""
        thresholds = {
            "overall": 75.0, "face_sharpness": 50.0,
            "face_contrast_high": 25.0, "face_contrast_low": 15.0,
            "noise": 20.0, "face_noise": 15.0,
        }

        face_detected = entry.get("face_detected") == "TRUE"
        overall_score = float(entry.get("overall_evaluation", 0.0))
        entry["accepted_flag"] = 0

        if face_detected:
            face_sharpness = float(entry.get("face_sharpness_score", 0.0))
            face_contrast = float(entry.get("face_contrast_score", 0.0))
            face_noise = float(entry.get("face_noise_score", 0.0))

            if overall_score >= thresholds["overall"]:
                if face_contrast >= thresholds["face_contrast_high"] and face_sharpness >= thresholds["face_sharpness"]:
                    entry["accepted_flag"] = 1
                elif face_contrast >= thresholds["face_contrast_low"] and face_sharpness >= thresholds["face_sharpness"]:
                    entry["accepted_flag"] = 2
                if face_noise > thresholds["face_noise"]:
                    entry["accepted_flag"] = min(entry["accepted_flag"], 1)
        else:
            sharpness = float(entry.get("sharpness_score", 0.0))
            contrast = float(entry.get("contrast_score", 0.0))
            noise = float(entry.get("noise_score", 0.0))

            if overall_score >= thresholds["overall"]:
                if contrast >= thresholds["face_contrast_high"] and sharpness >= thresholds["face_sharpness"]:
                    entry["accepted_flag"] = 1
                elif contrast >= thresholds["face_contrast_low"] and sharpness >= thresholds["face_sharpness"]:
                    entry["accepted_flag"] = 2
                if noise > thresholds["noise"]:
                    entry["accepted_flag"] = min(entry["accepted_flag"], 1)

    def rank_and_flag_top_entries(self, batch: List[Dict[str, str]]) -> None:
        """グループごとに上位35%にフラグ付け"""
        group_map: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for entry in batch:
            group_id = entry.get("group_id", "default")
            group_map[group_id].append(entry)

        for group_entries in group_map.values():
            group_entries.sort(key=lambda x: -float(x.get("overall_evaluation", 0.0)))
            threshold = max(1, int(len(group_entries) * 0.35))
            for i, entry in enumerate(group_entries):
                entry["flag"] = 1 if i < threshold else 0

    def cleanup(self) -> None:
        """一時ファイルをマージし、最終出力ファイルを生成"""
        super().cleanup()
        self.logger.info("Final output to be written from collected memory data...")

        merged_path = os.path.join(
            self.paths.get("output_data_dir", "./temp"),
            f"evaluation_ranking_{self.date}.csv"
        )

        if not self.output_data:  # ← 修正：temp_output_paths ではなく output_data
            self.logger.warning("No output data collected.")
            return

        fieldnames = ["file_name", "face_detected", "overall_evaluation", "flag", "accepted_flag"] + SCORE_TYPES

        sorted_data = sorted(
            self.output_data,  # ← 修正：self.output_data を使用
            key=lambda x: -float(x.get("overall_evaluation", 0.0))
        )

        with open(merged_path, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in sorted_data:
                writer.writerow({
                    "file_name": entry.get("file_name", "unknown"),
                    "face_detected": entry.get("face_detected", "FALSE"),
                    "overall_evaluation": round(float(entry.get("overall_evaluation", 0.0)), 2),
                    "flag": entry.get("flag", 0),
                    "accepted_flag": entry.get("accepted_flag", 0),
                    **{score: round(float(entry.get(score, 0.0) or 0.0), 2) for score in SCORE_TYPES},
                })

        self.logger.info(f"Final merged output written to {merged_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ポートレート評価バッチ処理")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file.")
    parser.add_argument("--date", type=str, help="Date for processing (format: YYYY-MM-DD).")
    args = parser.parse_args()

    processor = EvaluationRankBatchProcessor(
        config_path=args.config_path,
        date=args.date
    )
    processor.execute()
