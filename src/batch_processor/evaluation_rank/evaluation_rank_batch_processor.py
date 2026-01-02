#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from batch_framework.base_batch import BaseBatchProcessor
from batch_processor.evaluation_rank.scoring import EvaluationScorer
from batch_processor.evaluation_rank.acceptance import AcceptanceEngine, AcceptRules
from batch_processor.evaluation_rank.lightroom import apply_lightroom_fields
from batch_processor.evaluation_rank.writer import write_ranking_csv


# =========================
# utility
# =========================

def format_score(x: float) -> float:
    return round(float(x), 2)


def safe_float(value: Any) -> float:
    try:
        if value in ("", None):
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def parse_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"true", "1", "yes"}


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

        # BaseBatchProcessor の setup/process から参照される想定の属性をここで必ず持つ
        self.paths: Dict[str, str] = {}
        self.all_results: List[Dict[str, Any]] = []

        # scoring / calibration
        self.scorer = EvaluationScorer()
        self.calibration: Dict[str, float] = {}

        # acceptance / flag / reason（cleanupで使用）
        self.acceptance = AcceptanceEngine()

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
        # BaseBatchProcessor.setup が self.get_data() を呼ぶので先に用意
        self.paths.setdefault("evaluation_data_dir", "./temp")
        self.paths.setdefault("output_data_dir", "./output")

        # Base 側の契約:
        # setup() -> self.data = self.get_data() -> after_data_loaded(self.data)
        super().setup()

    def load_data(self) -> List[Dict[str, Any]]:
        """
        BaseBatchProcessor の新契約:
        - load_data(): 純I/O（副作用なし）
        - キャッシュは Base が握る（get_data() は Base 側）
        """
        input_csv = (
            Path(self.paths["evaluation_data_dir"])
            / f"evaluation_results_{self.date}.csv"
        )
        if not input_csv.exists():
            raise FileNotFoundError(input_csv)

        self.logger.info(f"Loading CSV: {input_csv}")
        with input_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        return rows

    def after_data_loaded(self, data: List[Dict]) -> None:
        """
        データロード後の副作用（calibration構築）をここに寄せる。
        Base.setup() から一度だけ呼ばれる前提。
        """
        self.scorer.build_calibration(data)
        self.calibration = dict(self.scorer.calibration)
        self.logger.info(
            "Calibration (P95) built: "
            + ", ".join([f"{k}={v:.3f}" for k, v in self.calibration.items()])
        )

    # -------------------------
    # batch
    # -------------------------

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for row in batch:
            try:
                face_detected = parse_bool(row.get("face_detected"))

                tech_score, tech_bd = self.scorer.technical_score(row)
                comp_score, comp_bd = self.scorer.composition_score(row)
                face_score, face_bd = self.scorer.face_score(row) if face_detected else (0.0, {})

                overall = self.scorer.overall_score(
                    face_detected=face_detected,
                    tech_score=tech_score,
                    face_score=face_score,
                    comp_score=comp_score,
                )

                out = dict(row)
                out["face_detected"] = face_detected
                out["overall_score"] = format_score(overall)

                # ===== breakdown（保持）=====
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
    # cleanup / ranking / output
    # -------------------------

    def cleanup(self) -> None:
        super().cleanup()

        # BaseBatchProcessor.process が all_results を持つ設計なので、ここで必ず取り込む
        rows = [
            r["row"]
            for r in self.all_results
            if r.get("status") == "success" and isinstance(r.get("row"), dict)
        ]

        if not rows:
            self.logger.warning("No successful rows to output.")
            return

        # ===== acceptance / category / accepted_reason / flag =====
        thresholds = self.acceptance.run(rows)
        portrait_thr = thresholds.get("portrait", 0.0)
        non_face_thr = thresholds.get("non_face", 0.0)

        # Lightroom 付与は CSV 直前に一括で（rows in-place）
        for r in rows:
            apply_lightroom_fields(r, keyword_max_len=90)

        # ===== 出力 =====
        output_csv = (
            Path(self.paths["output_data_dir"])
            / f"evaluation_ranking_{self.date}.csv"
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        base_columns = [
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

        # face_detected 表記だけ互換（TRUE/FALSE）を rows に反映してから書く
        for r in rows:
            r["face_detected"] = "TRUE" if bool(r.get("face_detected")) else "FALSE"

        columns = write_ranking_csv(
            output_csv=output_csv,
            rows=rows,
            base_columns=base_columns,
        )


        self.logger.info(f"Output written: {output_csv}")
        self.logger.info(
            f"Accepted thresholds: portrait(P{self.acceptance.rules.portrait_percentile}={portrait_thr:.2f}), "
            f"non_face(P{self.acceptance.rules.non_face_percentile}={non_face_thr:.2f})"
        )


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
