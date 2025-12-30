#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

from batch_framework.base_batch import BaseBatchProcessor


# =========================
# utility
# =========================

def format_score(x: float) -> float:
    return round(x, 2)


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def normalize(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(value / max_value, 1.0))


def normalize_inverse(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return max(0.0, min(1.0 - (value / max_value), 1.0))


def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"true", "1", "yes"}


def weighted_contribution(value, max_value, weight, inverse=False):
    if inverse:
        norm = normalize_inverse(value, max_value)
    else:
        norm = normalize(value, max_value)
    return norm * weight


# =========================
# parameters
# =========================

TECH_METRIC_MAX = {
    "blurriness": 1.0,
    "sharpness": 50.0,
    "noise": 30.0,
    "local_sharpness": 50.0,
}

FACE_METRIC_MAX = {
    "sharpness": 50.0,
    "contrast": 20.0,
    "noise": 30.0,
    "local_sharpness": 50.0,
    "local_contrast": 20.0,
}

TECH_WEIGHTS = {
    "sharpness": 0.30,
    "local_sharpness": 0.30,
    "noise": 0.25,
    "blurriness": 0.15,
}

FACE_WEIGHTS = {
    "sharpness": 0.30,
    "contrast": 0.15,
    "noise": 0.15,
    "local_sharpness": 0.25,
    "local_contrast": 0.15,
}

COMPOSITION_WEIGHTS = {
    "composition_rule_based_score": 0.30,
    "face_position_score": 0.25,
    "framing_score": 0.25,
    "face_direction_score": 0.10,
    "eye_contact_score": 0.10,
}


# =========================
# processor
# =========================

class EvaluationRankBatchProcessor(BaseBatchProcessor):

    def __init__(
        self,
        config_path: str,
        max_workers: int = 1,
        date: Optional[str] = None,
    ):
        super().__init__(config_path=config_path, max_workers=max_workers)
        self.date = self._parse_date(date)
        self.paths: dict[str, str] = {}
        self.all_results: list[dict] = []

    def _parse_date(self, date_str: Optional[str]) -> str:
        if date_str:
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                self.logger.warning(f"Invalid date '{date_str}', fallback to today")
        return datetime.datetime.now().strftime("%Y-%m-%d")

    # -------------------------
    # setup / data
    # -------------------------

    def setup(self):
        self.paths.setdefault("evaluation_data_dir", "./temp")
        self.paths.setdefault("output_data_dir", "./output")
        super().setup()

    def get_data(self) -> list[dict]:
        input_csv = (
            Path(self.paths["evaluation_data_dir"])
            / f"evaluation_results_{self.date}.csv"
        )
        if not input_csv.exists():
            raise FileNotFoundError(input_csv)

        with input_csv.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # -------------------------
    # batch
    # -------------------------

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []

        for row in batch:
            try:
                face_detected = parse_bool(row.get("face_detected"))

                tech_score, tech_bd = self._technical_score(row)
                comp_score, comp_bd = self._composition_score(row)
                face_score, face_bd = self._face_score(row) if face_detected else (0.0, {})

                if face_detected:
                    overall = (
                        0.45 * face_score
                        + 0.35 * tech_score
                        + 0.20 * comp_score
                    )
                else:
                    overall = (
                        0.55 * tech_score
                        + 0.45 * comp_score
                    )

                out = dict(row)
                out["face_detected"] = face_detected
                out["overall_score"] = format_score(overall)

                # ===== ④ ここが追加 =====
                out["score_technical"] = format_score(tech_score)
                out["score_face"] = format_score(face_score)
                out["score_composition"] = format_score(comp_score)

                for k, v in tech_bd.items():
                    out[f"contrib_tech_{k}"] = format_score(v * 100)

                for k, v in face_bd.items():
                    out[f"contrib_face_{k}"] = format_score(v * 100)

                for k, v in comp_bd.items():
                    out[f"contrib_comp_{k}"] = format_score(v * 100)

                results.append({
                    "status": "success",
                    "score": out["overall_score"],
                    "row": out,
                })

            except Exception as e:
                self.logger.exception("Evaluation failed")
                results.append({
                    "status": "failure",
                    "score": 0.0,
                    "row": row,
                    "error": str(e),
                })

        return results

    # -------------------------
    # scoring (with breakdown)
    # -------------------------

    def _technical_score(self, row):
        bd = {
            "sharpness": weighted_contribution(
                safe_float(row.get("sharpness_score")),
                TECH_METRIC_MAX["sharpness"],
                TECH_WEIGHTS["sharpness"],
            ),
            "local_sharpness": weighted_contribution(
                safe_float(row.get("local_sharpness_score")),
                TECH_METRIC_MAX["local_sharpness"],
                TECH_WEIGHTS["local_sharpness"],
            ),
            "noise": weighted_contribution(
                safe_float(row.get("noise_score")),
                TECH_METRIC_MAX["noise"],
                TECH_WEIGHTS["noise"],
                inverse=True,
            ),
            "blurriness": weighted_contribution(
                safe_float(row.get("blurriness_score")),
                TECH_METRIC_MAX["blurriness"],
                TECH_WEIGHTS["blurriness"],
                inverse=True,
            ),
        }
        return sum(bd.values()) * 100.0, bd

    def _face_score(self, row):
        bd = {
            "sharpness": weighted_contribution(
                safe_float(row.get("face_sharpness_score")),
                FACE_METRIC_MAX["sharpness"],
                FACE_WEIGHTS["sharpness"],
            ),
            "contrast": weighted_contribution(
                safe_float(row.get("face_contrast_score")),
                FACE_METRIC_MAX["contrast"],
                FACE_WEIGHTS["contrast"],
            ),
            "noise": weighted_contribution(
                safe_float(row.get("face_noise_score")),
                FACE_METRIC_MAX["noise"],
                FACE_WEIGHTS["noise"],
                inverse=True,
            ),
            "local_sharpness": weighted_contribution(
                safe_float(row.get("face_local_sharpness_score")),
                FACE_METRIC_MAX["local_sharpness"],
                FACE_WEIGHTS["local_sharpness"],
            ),
            "local_contrast": weighted_contribution(
                safe_float(row.get("face_local_contrast_score")),
                FACE_METRIC_MAX["local_contrast"],
                FACE_WEIGHTS["local_contrast"],
            ),
        }
        return sum(bd.values()) * 100.0, bd

    def _composition_score(self, row):
        bd = {
            k: weighted_contribution(
                safe_float(row.get(k)),
                1.0,
                COMPOSITION_WEIGHTS[k],
            )
            for k in COMPOSITION_WEIGHTS
        }
        return sum(bd.values()) * 100.0, bd

    # -------------------------
    # ranking & output（既存通り）
    # -------------------------

    def cleanup(self) -> None:
        super().cleanup()

        rows = [
            r["row"]
            for r in self.all_results
            if r.get("status") == "success"
        ]

        if not rows:
            self.logger.warning("No successful rows to output.")
            return

        # ===== ranking =====
        rows.sort(
            key=lambda r: safe_float(r.get("overall_score")),
            reverse=True
        )

        total = len(rows)
        top_ratio = 0.35
        top_n = max(1, int(total * top_ratio))

        # ===== flag / accepted_flag / category を付与 =====
        for i, r in enumerate(rows):
            face_detected = bool(r.get("face_detected"))

            r["category"] = "portrait" if face_detected else "non_face"
            r["flag"] = 1 if i < top_n else 0

            # accepted_flag は category ごとに緩めに
            if face_detected:
                r["accepted_flag"] = 1 if i < min(5, int(total * 0.30)) else 0
            else:
                r["accepted_flag"] = 1 if i < min(3, int(total * 0.20)) else 0

        output_csv = (
            Path(self.paths["output_data_dir"])
            / f"evaluation_ranking_{self.date}.csv"
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "file_name",
            "face_detected",
            "category",
            "overall_score",
            "flag",
            "accepted_flag",
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

        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for r in rows:
                r_out = dict(r)
                r_out["face_detected"] = (
                    "TRUE" if r_out.get("face_detected") else "FALSE"
                )
                writer.writerow({k: r_out.get(k) for k in columns})

        self.logger.info(f"Output written: {output_csv}")

# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/evaluation_rank.yaml")
    parser.add_argument("--date", required=True)
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    processor = EvaluationRankBatchProcessor(
        config_path=args.config_path,
        max_workers=args.max_workers,
        date=args.date,
    )
    processor.execute()


if __name__ == "__main__":
    main()
