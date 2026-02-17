#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from photo_insight.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.batch_processor.evaluation_rank.acceptance import AcceptanceEngine
from photo_insight.batch_processor.evaluation_rank.analysis.rejected_reason_stats import (
    RejectedReasonAnalyzer,
    write_rejected_reason_summary_csv,
)
from photo_insight.batch_processor.evaluation_rank.contract import validate_input_contract
from photo_insight.batch_processor.evaluation_rank.lightroom import apply_lightroom_fields
from photo_insight.batch_processor.evaluation_rank.scoring import (
    EvaluationScorer,
    apply_half_closed_penalty_to_expression,
    half_closed_eye_penalty_proxy,
    parse_gaze_y,
    score01,
)
from photo_insight.batch_processor.evaluation_rank.writer import write_ranking_csv
from photo_insight.batch_processor.evaluation_rank.analysis.provisional_vs_accepted import (
    build_provisional_vs_accepted_summary,
    write_provisional_vs_accepted_summary_csv,
)
from photo_insight.evaluators.common.grade_contract import GRADE_ENUM, normalize_eval_status, score_to_grade
from photo_insight.batch_processor.evaluation_rank.provisional import apply_provisional_top_percent



# =========================
# utility
# =========================

def format_score(x: Any) -> float:
    """表示用丸め（CSVの見た目だけ整える）"""
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def parse_bool(v: Any) -> bool:
    """CSV の TRUE/FALSE/1/0/yes/no などを bool 化。"""
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return False


def _as_scorepack(x: Any) -> Tuple[float, Dict[str, float]]:
    """
    Scorer が ScorePack(.score/.breakdown) を返す版でも、
    旧実装の float を返す版でも落ちないようにする互換層。
    """
    if hasattr(x, "score"):
        try:
            score = float(getattr(x, "score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        bd = getattr(x, "breakdown", None) or {}
        try:
            return score, dict(bd)
        except Exception:
            return score, {}
    try:
        return float(x), {}
    except Exception:
        return 0.0, {}


def inject_best_eye_features(row: Dict[str, Any]) -> None:
    """faces が壊れていても落とさず、best face の eye_* を row 直下へ入れる。"""
    faces = row.get("faces")
    if not isinstance(faces, list) or not faces:
        return

    best: Optional[Dict[str, Any]] = None
    best_conf = -1.0

    for f in faces:
        if not isinstance(f, dict):
            continue
        try:
            conf = float(f.get("confidence", 0.0) or 0.0)
        except Exception:
            conf = 0.0
        if conf > best_conf:
            best_conf = conf
            best = f

    if not isinstance(best, dict):
        return

    row["eye_closed_prob_best"] = best.get("eye_closed_prob", 0.0)
    row["eye_lap_var_best"] = best.get("eye_lap_var", 0.0)
    row["eye_patch_size_best"] = best.get("eye_patch_size", 0)


def _safe_parse_faces(value: Any) -> Tuple[List[Dict[str, Any]], str]:
    """
    faces列を「必ず List[Dict]」に復元する。
    戻り値: (faces_list, parse_status_reason)
    """
    def _coerce_face_dict(x: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(x, dict):
            return None
        d = dict(x)
        for key in ("box", "bbox"):
            if key in d and isinstance(d[key], str):
                s = d[key].strip()
                if s:
                    try:
                        d[key] = json.loads(s)
                    except Exception:
                        try:
                            d[key] = ast.literal_eval(s)
                        except Exception:
                            pass
        return d

    def _coerce_face_list(obj: Any) -> List[Dict[str, Any]]:
        if obj is None:
            return []
        if isinstance(obj, dict) and "faces" in obj:
            obj = obj.get("faces")

        if isinstance(obj, list):
            out: List[Dict[str, Any]] = []
            for it in obj:
                d = _coerce_face_dict(it)
                if d is not None:
                    out.append(d)
            return out

        if isinstance(obj, dict):
            d = _coerce_face_dict(obj)
            return [d] if d is not None else []
        return []

    if isinstance(value, list):
        return _coerce_face_list(value), "faces_parse:ok:list"

    if value in ("", None):
        return [], "faces_parse:empty"

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return [], "faces_parse:empty_str"

        try:
            obj = json.loads(s)
            return _coerce_face_list(obj), "faces_parse:ok:json"
        except Exception:
            pass

        try:
            obj = ast.literal_eval(s)
            return _coerce_face_list(obj), "faces_parse:ok:pyrepr"
        except Exception:
            return [], "faces_parse:fail:unparsed_str"

    return [], f"faces_parse:fail:type={type(value).__name__}"


def _normalize_grade_value(grade: Any) -> Optional[str]:
    """grade を contract に寄せる（最低限の揺れ吸収）。"""
    if grade in ("", None):
        return None
    s = str(grade).strip().lower()
    return s if s in GRADE_ENUM else None


def _ensure_grade_by_score(row: Dict[str, Any], metric: str, *, prefix: str = "") -> None:
    """{metric}_grade が欠損/不正な場合のみ、{metric}_score から grade を補完する。"""
    gk = f"{prefix}{metric}_grade"
    sk = f"{prefix}{metric}_score"

    g = _normalize_grade_value(row.get(gk))
    if g is not None:
        row[gk] = g
        return

    if sk in row:
        row[gk] = score_to_grade(row.get(sk))
    else:
        row[gk] = None


def _normalize_status_key(row: Dict[str, Any], key: str) -> None:
    """eval_status / *_status を grade_contract に正規化する（key が無いなら何もしない）。"""
    if key in row:
        row[key] = normalize_eval_status(row.get(key))


def _normalize_row_inplace(row: Dict[str, Any]) -> None:
    """
    evaluation_rank 側の “入口正規化”。
    """
    faces_list, faces_reason = _safe_parse_faces(row.get("faces"))
    row["faces"] = faces_list
    row["faces_parse_reason"] = faces_reason  # デバッグ用（契約外なら writer で落とす想定）

    row["face_detected"] = parse_bool(row.get("face_detected"))
    row["full_body_detected"] = parse_bool(row.get("full_body_detected"))

    for k in ("pose_score", "full_body_cut_risk"):
        if k in row:
            row[k] = safe_float(row.get(k), 0.0)

    for k in list(row.keys()):
        if isinstance(k, str) and k.endswith("_eval_status"):
            _normalize_status_key(row, k)

    _normalize_status_key(row, "composition_status")
    _normalize_status_key(row, "face_composition_status")

    comp_status = str(row.get("composition_status") or "").strip().lower()

    src = row.get("main_subject_center_source")
    src_reason = ""
    if isinstance(src, str) and src.strip():
        s_low = src.strip().lower()
        if s_low.startswith("invalid"):
            parts = [p.strip() for p in s_low.split(",") if p.strip()]
            row["main_subject_center_status"] = parts[0] if parts else "invalid"
            row["main_subject_center_source_parsed"] = parts[1] if len(parts) >= 2 else None
            src_reason = f"center_calc_failed:{row.get('main_subject_center_source_parsed') or 'unknown'}"
            row["main_subject_center_invalid_reason"] = src_reason
        else:
            row.setdefault("main_subject_center_status", None)
            row.setdefault("main_subject_center_source_parsed", None)
            row.setdefault("main_subject_center_invalid_reason", "")
    else:
        row.setdefault("main_subject_center_status", None)
        row.setdefault("main_subject_center_source_parsed", None)
        row.setdefault("main_subject_center_invalid_reason", "")

    if comp_status == "invalid":
        row["composition_invalid_reason"] = src_reason or str(row.get("composition_invalid_reason") or "unknown")
    else:
        row.setdefault("composition_invalid_reason", "")

    for m in ("sharpness", "blurriness", "contrast", "noise", "exposure", "expression", "face_blurriness", "face_contrast"):
        if m.startswith("face_"):
            continue
        _ensure_grade_by_score(row, m, prefix="")

    _ensure_grade_by_score(row, "blurriness", prefix="face_")
    _ensure_grade_by_score(row, "contrast", prefix="face_")
    _ensure_grade_by_score(row, "noise", prefix="face_")
    _ensure_grade_by_score(row, "exposure", prefix="face_")

    row["expression_grade"] = _normalize_grade_value(row.get("expression_grade")) or row.get("expression_grade")


# =========================
# processor
# =========================

class EvaluationRankBatchProcessor(BaseBatchProcessor):
    def __init__(
        self,
        config_path: Optional[str] = None,
        max_workers: int = 1,
        date: Optional[str] = None,
        # ===== Config DI (Baseへ委譲) =====
        config_env: Optional[str] = None,
        config_paths: Optional[List[str]] = None,
        resolver: Any = None,
        loader: Any = None,
        watch_factory: Any = None,
        list_policy: str = "replace",
        strict_missing: bool = True,
        auto_load: bool = True,
    ):
        super().__init__(
            config_path=config_path,
            config_env=config_env,
            config_paths=config_paths,
            max_workers=max_workers,
            logger=None,
            resolver=resolver,
            loader=loader,
            watch_factory=watch_factory,
            list_policy=list_policy,
            strict_missing=strict_missing,
            auto_load=auto_load,
        )

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
        """
        paths の初期値を config から埋める。
        - evaluation_data_dir: evaluation_results_YYYY-MM-DD.csv がある場所
        - output_data_dir: evaluation_ranking_YYYY-MM-DD.csv を出す場所
        """
        cfg = self.config_manager.get_config() or {}

        # 互換: batch/evaluation_rank のどちらかに置いても拾えるようにする
        rank_cfg = (cfg.get("evaluation_rank") or cfg.get("batch_processor") or {}) if isinstance(cfg, dict) else {}

        eval_dir = (
            rank_cfg.get("evaluation_data_dir")
            or (cfg.get("paths", {}) or {}).get("evaluation_data_dir")
            or "./temp"
        )
        out_dir = (
            rank_cfg.get("output_data_dir")
            or (cfg.get("paths", {}) or {}).get("output_data_dir")
            or "./output"
        )

        # project_root 基準に寄せる（相対パス事故を減らす）
        pr = Path(getattr(self, "project_root", "."))
        eval_dir_p = (Path(eval_dir) if Path(eval_dir).is_absolute() else (pr / eval_dir)).resolve()
        out_dir_p = (Path(out_dir) if Path(out_dir).is_absolute() else (pr / out_dir)).resolve()

        self.paths["evaluation_data_dir"] = str(eval_dir_p)
        self.paths["output_data_dir"] = str(out_dir_p)

        super().setup()

    def load_data(self) -> List[Dict[str, Any]]:
        """
        BaseBatchProcessor 契約:
        - load_data(): 純I/O（副作用なし）
        """
        input_csv = Path(self.paths["evaluation_data_dir"]) / f"evaluation_results_{self.date}.csv"
        if not input_csv.exists():
            raise FileNotFoundError(input_csv)

        self.logger.info(f"Loading CSV: {input_csv}")
        with input_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            validate_input_contract(header=header, csv_path=input_csv)
            rows = list(reader)
        return rows

    def after_data_loaded(self, data: List[Dict[str, Any]]) -> None:
        """
        データロード後の副作用（calibration構築）をここに寄せる。
        """
        try:
            for row in data:
                if isinstance(row, dict):
                    _normalize_row_inplace(row)
        except Exception:
            self.logger.exception("Row normalization failed in after_data_loaded (continue).")

        if hasattr(self.scorer, "build_calibration"):
            try:
                self.scorer.build_calibration(data)
                self.calibration = dict(getattr(self.scorer, "calibration", {}) or {})
                if self.calibration:
                    self.logger.info(
                        "Calibration built: "
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
        results: List[Dict[str, Any]] = []

        for row in batch:
            try:
                out = dict(row) if isinstance(row, dict) else {}

                _normalize_row_inplace(out)

                debug_pitch = 0.0
                debug_gaze_y = None
                debug_eye = 0.0
                debug_expr = 0.0
                half_pen = 0.0
                expr_eff = 0.0

                tech_sp = self.scorer.technical_score(out)
                face_sp = self.scorer.face_score(out)
                comp_sp = self.scorer.composition_score(out)

                tech_score, tech_bd = _as_scorepack(tech_sp)
                face_score, face_bd = _as_scorepack(face_sp)
                comp_score, comp_bd = _as_scorepack(comp_sp)

                overall = self.scorer.overall_score(out, tech_sp, face_sp, comp_sp)

                try:
                    debug_pitch = safe_float(out.get("pitch"), 0.0)
                    debug_gaze_y = parse_gaze_y(out.get("gaze"))
                    debug_eye = score01(out, "eye_contact_score", default=0.0)
                    debug_expr = score01(out, "expression_score", default=0.0)

                    expr = score01(out, "expression_score", default=0.0)
                    half_pen = float(half_closed_eye_penalty_proxy(out))
                    expr_eff = float(apply_half_closed_penalty_to_expression(expr, half_pen))
                except Exception:
                    pass

                out["debug_pitch"] = debug_pitch
                out["debug_gaze_y"] = "" if debug_gaze_y is None else float(debug_gaze_y)
                out["debug_eye_contact"] = debug_eye
                out["debug_expression"] = debug_expr
                out["debug_half_penalty"] = half_pen
                out["debug_expr_effective"] = expr_eff

                out["overall_score"] = float(overall)
                out["score_technical"] = float(tech_score)
                out["score_face"] = float(face_score)
                out["score_composition"] = float(comp_score)

                for k, v in (tech_bd or {}).items():
                    out[f"contrib_tech_{k}"] = safe_float(v) * 100.0
                for k, v in (face_bd or {}).items():
                    out[f"contrib_face_{k}"] = safe_float(v) * 100.0
                for k, v in (comp_bd or {}).items():
                    out[f"contrib_comp_{k}"] = safe_float(v) * 100.0

                results.append({"status": "success", "score": float(overall), "row": out})

            except Exception as e:
                self.logger.exception("Evaluation failed")
                results.append({"status": "failure", "score": 0.0, "row": row, "error": str(e)})

        return results


    def _log_thresholds(self, thresholds: dict[str, float]) -> None:
        try:
            portrait_thr = float(thresholds.get("portrait", 0.0) or 0.0)
            non_face_thr = float(thresholds.get("non_face", 0.0) or 0.0)

            rules = getattr(self.acceptance, "rules", None)
            portrait_p = getattr(rules, "portrait_percentile", None)
            non_face_p = getattr(rules, "non_face_percentile", None)

            # 互換（過去のフィールド名）
            if portrait_p is None:
                portrait_p = getattr(rules, "portrait_p", None) or getattr(rules, "portrait_accept_percentile", None)
            if non_face_p is None:
                non_face_p = getattr(rules, "non_face_p", None) or getattr(rules, "non_face_accept_percentile", None)

            portrait_p_txt = "?" if portrait_p is None else str(portrait_p)
            non_face_p_txt = "?" if non_face_p is None else str(non_face_p)

            self.logger.info(
                f"Accepted thresholds: portrait(P{portrait_p_txt}={portrait_thr:.2f}), "
                f"non_face(P{non_face_p_txt}={non_face_thr:.2f})"
            )
        except Exception as e:
            self.logger.warning(f"Failed to log thresholds: {e}")


    # -------------------------
    # cleanup / ranking / output
    # -------------------------

    def cleanup(self) -> None:
        try:
            rows = self._collect_success_rows()
            if not rows:
                self.logger.warning("No successful rows to output.")
                return

            self._normalize_rows(rows)
            self._inject_eye_features(rows)

            thresholds = self.acceptance.run(rows)

            self._apply_lightroom(rows)
            self._format_rows_for_output(rows)

            # 706-1
            self._apply_provisional_top_percent(rows)

            output_csv, columns = self._write_ranking(rows)

            # 706-3 Step1（NEW）
            self._write_provisional_vs_accepted_summary(rows)

            # rejected_reason（既存）
            self._write_rejected_reason_summary(rows)

            self._log_thresholds(thresholds)
            self.logger.info(f"Columns written: {len(columns)} cols")

        finally:
            super().cleanup()


    def _collect_success_rows(self) -> list[dict[str, Any]]:
        return [
            r["row"]
            for r in self.all_results
            if r.get("status") == "success" and isinstance(r.get("row"), dict)
        ]


    def _normalize_rows(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            _normalize_row_inplace(r)


    def _inject_eye_features(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            inject_best_eye_features(r)


    def _apply_lightroom(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            apply_lightroom_fields(r, keyword_max_len=90)


    def _format_rows_for_output(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
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


    def _write_ranking(self, rows: list[dict[str, Any]]) -> tuple[Path, list[str]]:
        output_csv = Path(self.paths["output_data_dir"]) / f"evaluation_ranking_{self.date}.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        columns = write_ranking_csv(output_csv=output_csv, rows=rows, sort_for_ranking=True)
        self.logger.info(f"Output written: {output_csv}")
        return output_csv, columns


    def _write_provisional_vs_accepted_summary(self, rows: list[dict[str, Any]]) -> None:
        try:
            out_dir = Path(self.paths["output_data_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"provisional_vs_accepted_summary_{self.date}.csv"

            summary, meta = build_provisional_vs_accepted_summary(rows)
            write_provisional_vs_accepted_summary_csv(summary, out_path)

            def _rate(n: int, d: int) -> float:
                return (float(n) / float(d)) if d > 0 else 0.0

            def _fmt(tag: str, r) -> str:
                total = int(getattr(r, "total", 0) or 0)

                accepted = int(getattr(r, "accepted", 0) or 0)
                provisional = int(getattr(r, "provisional", 0) or 0)
                overlap = int(getattr(r, "overlap", 0) or 0)
                accepted_not_top = int(getattr(r, "accepted_not_top", 0) or 0)
                top_not_accepted = int(getattr(r, "top_not_accepted", 0) or 0)

                acc_r = _rate(accepted, total)
                prov_r = _rate(provisional, total)
                ov_r = _rate(overlap, total)
                ant_r = _rate(accepted_not_top, total)
                tna_r = _rate(top_not_accepted, total)

                return (
                    f"{tag} total={total} "
                    f"accepted={accepted} ({acc_r:.3f}) "
                    f"prov={provisional} ({prov_r:.3f}) "
                    f"overlap={overlap} ({ov_r:.3f}) "
                    f"A_not_top={accepted_not_top} ({ant_r:.3f}) "
                    f"top_not_A={top_not_accepted} ({tna_r:.3f})"
                )

            if not summary:
                self.logger.info(f"[prov_vs_acc] wrote: {out_path} (empty)")
                return

            # 1) ALL/ALL（主ログ）
            all_row = next((r for r in summary if r.category == "ALL" and r.accept_group == "ALL"), None)
            if all_row:
                self.logger.info(
                    f"[prov_vs_acc] wrote: {out_path} "
                    f"({_fmt('ALL', all_row)}, groups={meta.get('groups')}, categories={meta.get('categories')})"
                )
            else:
                self.logger.info(
                    f"[prov_vs_acc] wrote: {out_path} (no ALL row, groups={meta.get('groups')}, categories={meta.get('categories')})"
                )

            # 2) portrait/ALL, non_face/ALL（あれば1行ずつ）
            portrait_all = next((r for r in summary if r.category == "portrait" and r.accept_group == "ALL"), None)
            if portrait_all:
                self.logger.info(f"[prov_vs_acc] {_fmt('portrait', portrait_all)}")

            non_face_all = next((r for r in summary if r.category == "non_face" and r.accept_group == "ALL"), None)
            if non_face_all:
                self.logger.info(f"[prov_vs_acc] {_fmt('non_face', non_face_all)}")

        except Exception as e:
            self.logger.warning(f"[prov_vs_acc] failed to write summary: {e}")


    def _write_rejected_reason_summary(self, rows: list[dict[str, Any]]) -> None:
        try:
            analyzer = RejectedReasonAnalyzer(
                alias_map={},
                unknown_label="unknown",
                keep_unknown_raw=False,
                # rejected_reason が無い/空でも accepted_reason を拾える版を使うならここも明示すると堅い
                reason_keys=("rejected_reason", "accepted_reason"),
            )
            summary, meta = analyzer.analyze(rows)

            out_dir = Path(self.paths["output_data_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "rejected_reason_summary.csv"

            write_rejected_reason_summary_csv(summary, out_path)

            self.logger.info(
                f"[rejected_reason] wrote: {out_path} "
                f"(total_rows={meta['total_rows']}, rejected={meta['total_rejected']}, "
                f"reasons={meta['unique_reasons']}, "
                f"missing_reason_rows={meta.get('missing_reason_rows')}, "
                f"reason_keys={meta.get('reason_keys')})"
            )

            if summary:
                top = ", ".join([f"{r.reason_code}:{r.count}" for r in summary[:3]])
                self.logger.info(f"[rejected_reason] top: {top}")


        except Exception as e:
            self.logger.warning(f"[rejected_reason] failed to write summary: {e}")


    def _apply_provisional_top_percent(self, rows: list[dict[str, Any]]) -> None:
        """
        #706-1 provisional top% flag
        rows に provisional_top_percent_flag / provisional_top_percent を in-place で付与し、
        運用で効く統計ログを出す。
        """
        try:
            cfg = self.config_manager.get_config() or {}
            rank_cfg = (
                (cfg.get("evaluation_rank") or cfg.get("batch_processor") or {})
                if isinstance(cfg, dict)
                else {}
            )
            prov_cfg = (rank_cfg.get("provisional_top_percent") or {}) if isinstance(rank_cfg, dict) else {}

            prov_enabled = bool(prov_cfg.get("enabled", False))
            prov_percent = prov_cfg.get("percent", 0)

            def _i01(v: Any) -> int:
                try:
                    return 1 if int(float(v)) != 0 else 0
                except Exception:
                    return 1 if str(v).strip().lower() in ("1", "true", "t", "yes", "y") else 0

            def _count_stats(items: list[dict[str, Any]]) -> tuple[int, int, int, int, int]:
                a = p = ov = 0
                for r in items:
                    af = _i01(r.get("accepted_flag", 0))
                    pf = _i01(r.get("provisional_top_percent_flag", 0))
                    a += af
                    p += pf
                    if af and pf:
                        ov += 1
                return a, p, ov, (a - ov), (p - ov)

            if prov_enabled:
                apply_provisional_top_percent(
                    records=rows,
                    percent=prov_percent,
                    score_key="overall_score",
                )

                try:
                    p = float(rows[0].get("provisional_top_percent", prov_percent) or 0.0)
                except Exception:
                    p = 0.0

                total = len(rows)
                accepted, provisional, overlap, accepted_not_top, top_not_accepted = _count_stats(rows)

                portraits = [r for r in rows if str(r.get("category") or "").strip().lower() == "portrait"]
                non_faces = [r for r in rows if str(r.get("category") or "").strip().lower() == "non_face"]

                pa, pp, pov, _, _ = _count_stats(portraits) if portraits else (0, 0, 0, 0, 0)
                na, np_, nov, _, _ = _count_stats(non_faces) if non_faces else (0, 0, 0, 0, 0)

                self.logger.info(
                    f"[provisional_top_percent] enabled p={p:.1f} k={provisional}/{total} "
                    f"accepted={accepted} overlap={overlap} "
                    f"accepted_not_top={accepted_not_top} top_not_accepted={top_not_accepted} "
                    f"| portrait(k={pp}/{len(portraits)} acc={pa} ov={pov}) "
                    f"non_face(k={np_}/{len(non_faces)} acc={na} ov={nov})"
                )
            else:
                total = len(rows)
                accepted = sum(_i01(r.get("accepted_flag", 0)) for r in rows)
                self.logger.info(f"[provisional_top_percent] disabled (accepted={accepted}/{total})")

        except Exception:
            self.logger.exception("Failed to apply provisional_top_percent (continue).")


# =========================
# CLI
# =========================

def _resolve_default_rank_yaml(project_root: str) -> str:
    """
    既存運用: config/evaluation_rank.yaml があるならそれをデフォルトにする。
    ただし存在しない場合は None 相当でBase/ConfigManagerに委譲。
    """
    p = (Path(project_root) / "config" / "evaluation_rank.yaml").resolve()
    return str(p) if p.exists() else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=None,
        help="Optional. If omitted, ConfigManager uses CONFIG_ENV / defaults.",
    )
    parser.add_argument(
        "--config_env",
        default=None,
        help="Optional env name (prod/test). If omitted, CONFIG_ENV env-var may be used.",
    )
    parser.add_argument(
        "--config_paths",
        nargs="*",
        default=None,
        help="Optional explicit config file list (applied in order; supports extends).",
    )
    parser.add_argument("--date", required=True)
    parser.add_argument("--max_workers", type=int, default=1)
    args = parser.parse_args()

    # project_root は Base でも計算されるが、CLIデフォルト解決にだけ使う
    project_root = str((Path(__file__).resolve().parents[1]))

    # 互換: 旧CLIと同じ挙動を維持（rank専用yamlがあるならそれをデフォルトに）
    # ただし明示指定がある場合はそれを優先
    config_path = args.config_path
    if config_path is None:
        fallback = _resolve_default_rank_yaml(project_root)
        config_path = fallback or None

    processor = EvaluationRankBatchProcessor(
        config_path=config_path,
        config_env=args.config_env,
        config_paths=args.config_paths,
        max_workers=args.max_workers,
        date=args.date,
    )
    processor.execute()


if __name__ == "__main__":
    main()
