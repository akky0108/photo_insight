#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from batch_framework.base_batch import BaseBatchProcessor
from batch_processor.evaluation_rank.scoring import EvaluationScorer
from batch_processor.evaluation_rank.acceptance import AcceptanceEngine
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
# secondary accept (候補採用) 用の設定
# =========================

# thr からどれだけマージンをとるか (sec_thr = max(thr - margin, min_overall))
SECONDARY_MARGIN = 3.0        # 例: thr=56.1 → sec_thr = max(53.1, 60.0) = 60.0
SECONDARY_MIN_OVERALL = 60.0  # これ未満は secondary にも入れない
SECONDARY_MIN_FACE = 65.0     # score_face の最低ライン
SECONDARY_MIN_COMP = 45.0     # score_composition の最低ライン


def decide_secondary_accept(
    row: Dict[str, Any],
    thr: float,
    *,
    margin: float = SECONDARY_MARGIN,
    min_overall: float = SECONDARY_MIN_OVERALL,
    min_face: float = SECONDARY_MIN_FACE,
    min_comp: float = SECONDARY_MIN_COMP,
) -> tuple[bool, Optional[str]]:
    """
    一次採用には入らなかったが「人に渡せる候補」としてキープしたいカットを判定する。
    返り値: (secondary_accept_flag, secondary_accept_reason)
    """

    # すでに accepted_flag=1 なら secondary 対象外
    if row.get("accepted_flag"):
        return False, None

    overall = safe_float(row.get("overall_score", 0.0))
    score_face = safe_float(row.get("score_face", 0.0))
    score_comp = safe_float(row.get("score_composition", 0.0))

    # グループ毎の second 用しきい値
    secondary_thr = max(thr - margin, min_overall)

    if (
        overall >= secondary_thr
        and score_face >= min_face
        and score_comp >= min_comp
    ):
        reason = (
            f"SEC:{row.get('category', '')} "
            f"thr={thr:.2f} sec_thr={secondary_thr:.2f} "
            f"overall={overall:.2f} face={score_face:.2f} comp={score_comp:.2f}"
        )
        return True, reason

    return False, None

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
        """
        各行ごとに tech / face / comp / overall を計算し、
        rows に score_* / contrib_* / overall_score などを埋める。

        ※ 内部計算値は丸めず保持しておき、
           表示（accepted_reasonやCSV上）で丸める方針。
        """
        results: List[Dict[str, Any]] = []

        for row in batch:
            try:
                face_detected = parse_bool(row.get("face_detected"))
                # ★ full body 関連の情報を取得
                full_body_detected = parse_bool(row.get("full_body_detected"))
                pose_score = safe_float(row.get("pose_score"))
                full_body_cut_risk = safe_float(row.get("full_body_cut_risk"))


                tech_score, tech_bd = self.scorer.technical_score(row)
                comp_score, comp_bd = self.scorer.composition_score(row)
                face_score, face_bd = (
                    self.scorer.face_score(row) if face_detected else (0.0, {})
                )

                overall = self.scorer.overall_score(
                    face_detected=face_detected,
                    tech_score=tech_score,
                    face_score=face_score,
                    comp_score=comp_score,
                    full_body_detected=full_body_detected,
                    pose_score=pose_score,
                    full_body_cut_risk=full_body_cut_risk,
                )

                out = dict(row)
                out["face_detected"] = face_detected
                out["full_body_detected"] = full_body_detected

                # ===== scores（内部は生値で保持）=====
                out["overall_score"] = overall
                out["score_technical"] = tech_score
                out["score_face"] = face_score
                out["score_composition"] = comp_score

                # ===== breakdown（0..100スケールに揃えつつ生値保持）=====
                # technical_score / face_score / comp_score では sum(bd.values())*100 しているので
                # breakdown側も *100 してあげておく。
                for k, v in tech_bd.items():
                    out[f"contrib_tech_{k}"] = v * 100.0

                for k, v in face_bd.items():
                    out[f"contrib_face_{k}"] = v * 100.0

                for k, v in comp_bd.items():
                    out[f"contrib_comp_{k}"] = v * 100.0

                results.append(
                    {
                        "status": "success",
                        "score": overall,  # summarize用も生値
                        "row": out,
                    }
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

        # ===== secondary_accept_flag / secondary_accept_reason を付与 =====
        secondary_count = 0
        for r in rows:
            category = (r.get("category") or "").lower()

            # いまは portrait のみ secondary 対象とする
            if category == "portrait" and portrait_thr > 0.0:
                sec_flag, sec_reason = decide_secondary_accept(r, portrait_thr)
            else:
                sec_flag, sec_reason = (False, None)

            r["secondary_accept_flag"] = 1 if sec_flag else 0
            r["secondary_accept_reason"] = sec_reason or ""

            if sec_flag:
                secondary_count += 1

        self.logger.info(
            f"Secondary accepted (candidate) count: {secondary_count}"
        )

        # Lightroom 付与は CSV 直前に一括で（rows in-place）
        for r in rows:
            # overall_score は内部では生値なので、LR側で safe_float → score_to_rating でよしなにしてくれる
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
        ]

        # face_detected 表記だけ互換（TRUE/FALSE）を rows に反映してから書く
        for r in rows:
            r["face_detected"] = "TRUE" if bool(r.get("face_detected")) else "FALSE"
            # overall_score などはここで丸めてもOK（CSVの見た目を整える用途）
            if "overall_score" in r:
                r["overall_score"] = format_score(r["overall_score"])
            if "score_technical" in r:
                r["score_technical"] = format_score(r["score_technical"])
            if "score_face" in r:
                r["score_face"] = format_score(r["score_face"])
            if "score_composition" in r:
                r["score_composition"] = format_score(r["score_composition"])
            # contrib_* はそのままでも良いが、見やすさ重視ならここで丸めてもOK
            for k in list(r.keys()):
                if k.startswith("contrib_"):
                    r[k] = format_score(r[k])

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
