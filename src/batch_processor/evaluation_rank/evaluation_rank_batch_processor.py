#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from batch_framework.base_batch import BaseBatchProcessor
from batch_processor.evaluation_rank.scoring import EvaluationScorer
from batch_processor.evaluation_rank.acceptance import AcceptanceEngine
from batch_processor.evaluation_rank.lightroom import apply_lightroom_fields
from batch_processor.evaluation_rank.writer import write_ranking_csv

from batch_processor.evaluation_rank.scoring import (
    half_closed_eye_penalty_proxy,
    apply_half_closed_penalty_to_expression,
    parse_gaze_y,
    score01,
)

# =========================
# utility
# =========================

def format_score(x: float) -> float:
    return round(float(x), 2)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def parse_bool(v: Any) -> bool:
    """
    CSV の TRUE/FALSE/1/0/yes/no などを bool 化。
    """
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    # 数値っぽいものだけ拾う
    try:
        return bool(int(float(s)))
    except Exception:
        return False


def _as_scorepack(x: Any) -> Tuple[float, Dict[str, float]]:
    """
    Scorer が ScorePack(.score/.breakdown) を返す版でも、
    旧実装の float を返す版でも落ちないようにする互換層。
    """
    # ScorePack (score, breakdown)
    if hasattr(x, "score"):
        try:
            score = float(getattr(x, "score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        bd = getattr(x, "breakdown", None) or {}
        # breakdown は dict を想定
        try:
            return score, dict(bd)
        except Exception:
            return score, {}
    # float
    try:
        return float(x), {}
    except Exception:
        return 0.0, {}


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

        # BaseBatchProcessor の setup/process から参照される想定の属性
        self.paths: Dict[str, str] = {}
        self.all_results: List[Dict[str, Any]] = []

        # scoring / calibration
        self.scorer = EvaluationScorer()
        self.calibration: Dict[str, float] = {}

        # acceptance / flag / reason
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

    def setup(self) -> None:
        # BaseBatchProcessor.setup が self.get_data() を呼ぶので先に用意
        self.paths.setdefault("evaluation_data_dir", "./temp")
        self.paths.setdefault("output_data_dir", "./output")
        super().setup()

    def load_data(self) -> List[Dict[str, Any]]:
        """
        BaseBatchProcessor 契約:
        - load_data(): 純I/O（副作用なし）
        - キャッシュは Base が握る
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

    def after_data_loaded(self, data: List[Dict[str, Any]]) -> None:
        """
        データロード後の副作用（calibration構築）をここに寄せる。
        scorer 側に build_calibration が無い場合でも落とさない（将来/旧版互換）。
        """
        if hasattr(self.scorer, "build_calibration"):
            try:
                self.scorer.build_calibration(data)
                # scorer.calibration を持っている前提（無い場合は空）
                self.calibration = dict(getattr(self.scorer, "calibration", {}) or {})
                if self.calibration:
                    self.logger.info(
                        "Calibration (P95) built: "
                        + ", ".join([f"{k}={v:.3f}" for k, v in self.calibration.items()])
                    )
                else:
                    self.logger.info("Calibration built (empty).")
            except Exception:
                self.logger.exception("Calibration build failed. Continue without calibration.")
                self.calibration = {}
        else:
            self.logger.info("scorer.build_calibration not found. Continue without calibration.")
            self.calibration = {}

    # -------------------------
    # batch
    # -------------------------

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        各行ごとに tech / face / comp / overall を計算し、
        rows に score_* / contrib_* / overall_score などを埋める。
        """
        results: List[Dict[str, Any]] = []

        for row in batch:
            try:
                if row.get("file_name") == "DSC_7135.NEF":
                    self.logger.info(f"[debug] keys={sorted(list(row.keys()))[:60]} ... total={len(row.keys())}")
                    self.logger.info(f"[debug] pitch={row.get('pitch')} gaze={row.get('gaze')} expr={row.get('expression_score')} eye={row.get('eye_contact_score')}")

                out = dict(row)

                # debug値は先に必ず初期化（例外が起きても参照できるように）
                debug_pitch = 0.0
                debug_gaze_y = None
                debug_eye = 0.0
                debug_expr = 0.0
                half_pen = 0.0
                expr_eff = 0.0

                # 元CSV値が "TRUE"/"FALSE" 等でも扱えるように normalize
                face_detected = parse_bool(row.get("face_detected"))

                # 追加情報（あれば）
                full_body_detected = parse_bool(row.get("full_body_detected"))
                pose_score = safe_float(row.get("pose_score"))
                full_body_cut_risk = safe_float(row.get("full_body_cut_risk"))

                # --- scorer ---
                tech_sp = self.scorer.technical_score(row)
                face_sp = self.scorer.face_score(row)
                comp_sp = self.scorer.composition_score(row)

                tech_score, tech_bd = _as_scorepack(tech_sp)
                face_score, face_bd = _as_scorepack(face_sp)
                comp_score, comp_bd = _as_scorepack(comp_sp)

                # overall_score は scorer 側で ScorePack を前提にしている可能性があるので
                # 元のオブジェクト（tech_sp 等）を渡す
                overall = self.scorer.overall_score(row, tech_sp, face_sp, comp_sp)

                out["face_detected"] = face_detected
                out["full_body_detected"] = full_body_detected
                out["pose_score"] = pose_score
                out["full_body_cut_risk"] = full_body_cut_risk

                # --- debug (half-closed eyes proxy) ---
                try:
                    debug_pitch = safe_float(row.get("pitch"), 0.0)
                    debug_gaze_y = parse_gaze_y(row.get("gaze"))
                    debug_eye = score01(row, "eye_contact_score", default=0.0)
                    debug_expr = score01(row, "expression_score", default=0.0)

                    expr = score01(row, "expression_score", default=0.0)
                    half_pen = float(half_closed_eye_penalty_proxy(row))
                    expr_eff = float(apply_half_closed_penalty_to_expression(expr, half_pen))
                except Exception:
                    pass

                out["debug_pitch"] = debug_pitch
                out["debug_gaze_y"] = "" if debug_gaze_y is None else float(debug_gaze_y)
                out["debug_eye_contact"] = debug_eye
                out["debug_expression"] = debug_expr
                out["debug_half_penalty"] = half_pen
                out["debug_expr_effective"] = expr_eff


                # ===== scores（内部は生値で保持）=====
                out["overall_score"] = float(overall)
                out["score_technical"] = float(tech_score)
                out["score_face"] = float(face_score)
                out["score_composition"] = float(comp_score)

                # ===== breakdown（0..100スケールに揃える）=====
                for k, v in (tech_bd or {}).items():
                    out[f"contrib_tech_{k}"] = safe_float(v) * 100.0
                for k, v in (face_bd or {}).items():
                    out[f"contrib_face_{k}"] = safe_float(v) * 100.0
                for k, v in (comp_bd or {}).items():
                    out[f"contrib_comp_{k}"] = safe_float(v) * 100.0

                results.append(
                    {
                        "status": "success",
                        "score": float(overall),  # summarize用
                        "row": out,
                    }
                )

                if out.get("file_name") == "DSC_7135.NEF":
                    self.logger.info(
                        "[out-debug] "
                        f"debug_pitch={out.get('debug_pitch')} "
                        f"debug_gaze_y={out.get('debug_gaze_y')} "
                        f"debug_eye={out.get('debug_eye_contact')} "
                        f"debug_expr={out.get('debug_expression')} "
                        f"half={out.get('debug_half_penalty')} "
                        f"expr_eff={out.get('debug_expr_effective')}"
                    )

            except Exception as e:
                self.logger.exception("Evaluation failed")
                results.append(
                    {
                        "status": "failure",
                        "score": 0.0,
                        "row": row,
                        "error": str(e),
                    }
                )

        return results

    # -------------------------
    # cleanup / ranking / output
    # -------------------------

    def cleanup(self) -> None:
        """
        重要:
        - secondary 判定をここで別途やらない（AcceptanceEngine に統一）
        - Base の cleanup は最後に呼ぶ（出力後）
        """
        try:
            # BaseBatchProcessor.process が all_results を持つ設計なので、ここで必ず取り込む
            rows = [
                r["row"]
                for r in self.all_results
                if r.get("status") == "success" and isinstance(r.get("row"), dict)
            ]

            if not rows:
                self.logger.warning("No successful rows to output.")
                return

            # ===== acceptance / category / accepted_reason / flag を 1本化 =====
            thresholds = self.acceptance.run(rows)
            portrait_thr = thresholds.get("portrait", 0.0)
            non_face_thr = thresholds.get("non_face", 0.0)

            # ===== Lightroom 付与（accepted_reason 一本=A統一前提）=====
            for r in rows:
                r["accepted_flag"] = 1 if str(r.get("accepted_flag", "0")).strip() in {"1", "true", "TRUE"} else 0
                r["secondary_accept_flag"] = 1 if str(r.get("secondary_accept_flag", "0")).strip() in {"1", "true", "TRUE"} else 0
                r["flag"] = 1 if str(r.get("flag", "0")).strip() in {"1", "true", "TRUE"} else 0
                apply_lightroom_fields(r, keyword_max_len=90)

            # ===== 出力 =====
            output_csv = (
                Path(self.paths["output_data_dir"])
                / f"evaluation_ranking_{self.date}.csv"
            )
            output_csv.parent.mkdir(parents=True, exist_ok=True)

            base_columns = [
                "file_name",
                "group_id",
                "subgroup_id",
                "shot_type",
                "face_detected",
                "category",
                "overall_score",
                "flag",
                "accepted_flag",
                "secondary_accept_flag",
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
                # Debug
                "debug_pitch",
                "debug_gaze_y",
                "debug_eye_contact",
                "debug_expression",
                "debug_half_penalty",
                "debug_expr_effective",
                # debug end
                "score_composition",
                "score_face",
                "score_technical",
                "lr_keywords",
                "lr_rating",
                "lr_color_label",
                "lr_labelcolor_key",
                "lr_label_display",
                "accepted_reason",
            ]

            # face_detected 表記だけ互換（TRUE/FALSE）に寄せる + 表示用丸め
            for r in rows:
                r["face_detected"] = "TRUE" if bool(r.get("face_detected")) else "FALSE"

                # 見た目用の丸め（内部ロジックは既に終わっているのでここでOK）
                for k in (
                    "overall_score", "score_technical", "score_face", "score_composition",
                    "debug_pitch", "debug_gaze_y", "debug_eye_contact", "debug_expression",
                    "debug_half_penalty", "debug_expr_effective",
                ):
                    if k in r:
                        r[k] = format_score(safe_float(r.get(k)))

                for k in list(r.keys()):
                    if isinstance(k, str) and k.startswith("contrib_"):
                        r[k] = format_score(safe_float(r.get(k)))

            for r in rows:
                if r.get("file_name") == "DSC_7135.NEF":
                    self.logger.info(
                        "[prewrite] "
                        f"debug_pitch={r.get('debug_pitch')} "
                        f"debug_gaze_y={r.get('debug_gaze_y')} "
                        f"debug_eye={r.get('debug_eye_contact')} "
                        f"debug_expr={r.get('debug_expression')} "
                        f"half={r.get('debug_half_penalty')} "
                        f"expr_eff={r.get('debug_expr_effective')}"
                    )
                    break

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
        finally:
            # Base cleanup は最後に
            super().cleanup()


# =========================
# CLI
# =========================

def main() -> None:
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
