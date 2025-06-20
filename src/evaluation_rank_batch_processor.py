# evaluation_rank_batch_processor.py

import argparse
import csv
import datetime
import os
from typing import List, Dict, Optional
import yaml
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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

SCORE_TYPES = [
    "blurriness_score",
    "sharpness_score",
    "contrast_score",
    "noise_score",
    "local_sharpness_score",
    "local_contrast_score",
    "face_sharpness_score",
    "face_contrast_score",
    "face_noise_score",
    "face_local_sharpness_score",
    "face_local_contrast_score",
    "composition_rule_based_score",
    "face_position_score",
    "framing_score",
    "face_direction_score",
    "eye_contact_score",
]


def weighted_sum(values: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(values.get(k, 0.0) * weights.get(k, 0.0) for k in weights)


class EvaluationRankBatchProcessor(BaseBatchProcessor):

    def __init__(
        self,
        config_path: str,
        max_workers: int = 1,
        max_process_count: int = 5000,
        date: Optional[str] = None,
    ):
        super().__init__(
            config_path=config_path,
            max_workers=max_workers
        )
        self.date = self._parse_date(date)
        self.weights = {}
        self.paths = {}
        self.evaluation_data: List[Dict[str, str]] = []

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

        evaluation_file_path = os.path.join(
            self.paths.get("evaluation_data_dir", "./temp"),
            f"evaluation_results_{self.date}.csv",
        )
        if not os.path.exists(evaluation_file_path):
            raise FileNotFoundError(
                f"Evaluation data file not found: {evaluation_file_path}"
            )

        self.evaluation_data = self.load_evaluation_data(evaluation_file_path)

    def load_config(self, config_path: str) -> None:
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
        self.logger.info(f"Loading evaluation data from {file_path}.")
        with open(file_path, "r", encoding="utf-8") as csvfile:
            return [
                {**row, "face_detected": row.get("face_detected", "TRUE").upper()}
                for row in csv.DictReader(csvfile)
            ]

    def calculate_overall_evaluation(self, entry: Dict[str, str]) -> None:
        face_detected = entry.get("face_detected", "FALSE") == "TRUE"

        if face_detected:
            values = {
                "sharpness": float(entry.get("face_sharpness_score", 0.0)),
                "contrast": float(entry.get("face_contrast_score", 0.0)),
                "noise": float(entry.get("face_noise_score", 0.0)),
                "local_sharpness": float(entry.get("face_local_sharpness_score", 0.0)),
                "local_contrast": float(entry.get("face_local_contrast_score", 0.0)),
            }
            default_weights = FACE_WEIGHTS
            weight_key = "face"
        else:
            values = {
                "sharpness": float(entry.get("sharpness_score", 0.0)),
                "contrast": float(entry.get("contrast_score", 0.0)),
                "noise": float(entry.get("noise_score", 0.0)),
                "local_sharpness": float(entry.get("local_sharpness_score", 0.0)),
                "local_contrast": float(entry.get("local_contrast_score", 0.0)),
            }
            default_weights = GENERAL_WEIGHTS
            weight_key = "general"

        weights = self.weights.get(weight_key, default_weights)
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

    def assign_acceptance_flag(self, entry: Dict[str, str]) -> None:
        thresholds = {
            "overall": 75.0,
            "face_sharpness": 50.0,
            "face_contrast_high": 25.0,
            "face_contrast_low": 15.0,
            "noise": 20.0,
            "face_noise": 15.0,
        }

        face_detected = entry.get("face_detected") == "TRUE"
        overall_score = float(entry.get("overall_evaluation", 0.0))
        entry["accepted_flag"] = 0

        if face_detected:
            face_sharpness = float(entry.get("face_sharpness_score", 0.0))
            face_contrast = float(entry.get("face_contrast_score", 0.0))
            face_noise = float(entry.get("face_noise_score", 0.0))

            if overall_score >= thresholds["overall"]:
                if (
                    face_contrast >= thresholds["face_contrast_high"]
                    and face_sharpness >= thresholds["face_sharpness"]
                ):
                    entry["accepted_flag"] = 1
                elif (
                    face_contrast >= thresholds["face_contrast_low"]
                    and face_sharpness >= thresholds["face_sharpness"]
                ):
                    entry["accepted_flag"] = 2

                if face_noise > thresholds["face_noise"]:
                    entry["accepted_flag"] = min(entry["accepted_flag"], 1)
        else:
            sharpness = float(entry.get("sharpness_score", 0.0))
            contrast = float(entry.get("contrast_score", 0.0))
            noise = float(entry.get("noise_score", 0.0))

            if overall_score >= thresholds["overall"]:
                if (
                    contrast >= thresholds["face_contrast_high"]
                    and sharpness >= thresholds["face_sharpness"]
                ):
                    entry["accepted_flag"] = 1
                elif (
                    contrast >= thresholds["face_contrast_low"]
                    and sharpness >= thresholds["face_sharpness"]
                ):
                    entry["accepted_flag"] = 2

                if noise > thresholds["noise"]:
                    entry["accepted_flag"] = min(entry["accepted_flag"], 1)

    def process(self) -> None:
        self.logger.info("Processing started.")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self.calculate_overall_evaluation, self.evaluation_data)

        for entry in self.evaluation_data:
            self.assign_acceptance_flag(entry)

        group_map = defaultdict(list)
        for entry in self.evaluation_data:
            group_id = entry.get("group_id", "default")
            group_map[group_id].append(entry)

        for group_entries in group_map.values():
            group_entries.sort(key=lambda x: -float(x.get("overall_evaluation", 0.0)))
            threshold = max(1, int(len(group_entries) * 0.35))
            for i, entry in enumerate(group_entries):
                entry["flag"] = 1 if i < threshold else 0

        sorted_data = sorted(self.evaluation_data, key=lambda x: x.get("file_name", ""))
        output_file_path = os.path.join(
            self.paths.get("output_data_dir", "./temp"),
            f"evaluation_ranking_{self.date}.csv",
        )
        self.output_results(sorted_data, output_file_path)

        self.logger.info("Processing completed.")

    def _process_batch(self, batch=None):
        self.process()

    def get_data(self):
        return super().get_data()

    def output_results(
        self, sorted_data: List[Dict[str, str]], output_file_path: str
    ) -> None:
        self.logger.info(f"Outputting results to {output_file_path}.")
        with open(output_file_path, "w", encoding="utf-8", newline="") as csvfile:
            fieldnames = [
                "file_name",
                "face_detected",
                "overall_evaluation",
                "flag",
                "accepted_flag",
            ] + SCORE_TYPES
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in sorted_data:
                writer.writerow(
                    {
                        "file_name": entry.get("file_name", "unknown"),
                        "face_detected": entry.get("face_detected", "FALSE"),
                        "overall_evaluation": round(
                            float(entry.get("overall_evaluation", 0.0)), 2
                        ),
                        "flag": entry.get("flag", 0),
                        "accepted_flag": entry.get("accepted_flag", 0),
                        **{
                            score: round(float(entry.get(score, 0.0) or 0.0), 2)
                            for score in SCORE_TYPES
                        },
                    }
                )
        self.logger.info("Results output completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ポートレート評価バッチ処理")
    parser.add_argument(
        "--config_path", type=str, help="Path to the configuration file."
    )
    parser.add_argument(
        "--date", type=str, help="Date for processing (format: YYYY-MM-DD)."
    )
    args = parser.parse_args()

    processor = EvaluationRankBatchProcessor(
        config_path=args.config_path, date=args.date
    )
    processor.execute()
