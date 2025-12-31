#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import datetime
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from batch_framework.base_batch import BaseBatchProcessor


# =========================
# parameters
# =========================

# blurriness は 0..1 前提で固定が安定（P95にすると「ブレの少ない日」で効き過ぎることがある）
BLURRINESS_MAX = 1.0

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

# accepted_flag（分布ベース）
ACCEPT_RULES = {
    "portrait": {"percentile": 70, "max_accept": 5},
    "non_face": {"percentile": 80, "max_accept": 3},
}

# flag（従来通り：全体上位比率）
TOP_FLAG_RATIO = 0.35

# Lightroom（日本語ラベルセット想定）: 表示名マップ
LR_LABEL_DISPLAY_JA = {
    "red": "レッド",
    "yellow": "イエロー",
    "green": "グリーン",
    "blue": "ブルー",
    "purple": "パープル",
    "none": "",
    "": "",
}


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


def weighted_contribution(value: float, max_value: float, weight: float, inverse: bool = False) -> float:
    if inverse:
        norm = normalize_inverse(value, max_value)
    else:
        norm = normalize(value, max_value)
    return norm * weight


def percentile(values: List[float], p: float) -> float:
    """
    numpy無しpercentile（線形補間）。
    p: 0-100
    """
    xs = sorted([float(x) for x in values if x is not None])
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]

    p = max(0.0, min(100.0, float(p)))
    k = (n - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def build_accepted_reason(
    *,
    category: str,
    rank: int,
    max_accept: int,
    threshold: float,
    overall: float,
    accepted_flag: int,
    score_technical: Optional[float],
    score_face: Optional[float],
    score_composition: Optional[float],
    contrib_tech: Dict[str, Any],
    contrib_face: Dict[str, Any],
    contrib_comp: Dict[str, Any],
    top_n: int = 3,
) -> str:
    """
    accepted_reason を 1列にまとめる。
    - accepted_flag=1: contrib 上位Nだけで短文化
    - accepted_flag=0: フル（デバッグ用に残す）
    """
    def _to_float(d: Dict[str, Any]) -> Dict[str, float]:
        out = {}
        for k, v in d.items():
            fv = safe_float(v)
            out[k] = fv
        return out

    def _top_items(d: Dict[str, float], n: int) -> List[tuple[str, float]]:
        items = [(k, v) for k, v in d.items() if v is not None]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]

    parts = []
    parts.append(f"{category}")
    parts.append(f"rank={rank}/{max_accept}")
    parts.append(f"thr={format_score(threshold)}")
    parts.append(f"overall={format_score(overall)}")

    # score breakdown
    s = []
    if score_technical is not None:
        s.append(f"tech={format_score(score_technical)}")
    if score_face is not None:
        s.append(f"face={format_score(score_face)}")
    if score_composition is not None:
        s.append(f"comp={format_score(score_composition)}")
    if s:
        parts.append("(" + " ".join(s) + ")")

    t = _to_float(contrib_tech)
    f = _to_float(contrib_face)
    c = _to_float(contrib_comp)

    if accepted_flag == 1:
        merged = {}
        for k, v in t.items():
            merged[f"tech.{k}"] = v
        for k, v in f.items():
            merged[f"face.{k}"] = v
        for k, v in c.items():
            merged[f"comp.{k}"] = v

        top = _top_items(merged, top_n)
        top_txt = ", ".join([f"{k}={format_score(v)}" for k, v in top]) if top else ""

        # ✅ accepted_flag=1 は “短文” に寄せる（必要情報だけ）
        return (
            f"{category} rank={rank}/{max_accept} thr={format_score(threshold)} "
            f"overall={format_score(overall)} top={top_txt}"
        )

    # ===== フル版（accepted_flag=0 のときだけ）=====
    def pack_contrib(label: str, d: Dict[str, float]) -> str:
        if not d:
            return ""
        items = [(k, v) for k, v in d.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return f"{label}[" + ", ".join([f"{k}={format_score(v)}" for k, v in items]) + "]"

    for x in (pack_contrib("tech", t), pack_contrib("face", f), pack_contrib("comp", c)):
        if x:
            parts.append(x)

    return " | ".join(parts)


def score_to_rating(overall: float) -> int:
    """
    overall_score (0–100) → Lightroom 星評価 (0–5)

    運用意図:
    ★★★★★ (5): そのまま納品・公開・代表作候補（確実に残す）
    ★★★★☆ (4): 非常に良い。用途次第で採用
    ★★★☆☆ (3): 記録・素材として有用（削らない）
    ★★☆☆☆ (2): 微妙だが状況証拠として残す可能性あり
    ★☆☆☆☆ (1): 基本不要だが誤検出・検証用
    ☆☆☆☆☆ (0): 自動削除候補

    ※ 閾値は「日ごとの分布」よりも
       人間の最終判断基準として安定させる目的で固定値。
    """
    if overall >= 85: return 5
    if overall >= 75: return 4
    if overall >= 65: return 3
    if overall >= 55: return 2
    if overall >= 45: return 1
    return 0


def choose_color_label(category: str, accepted_flag: int, rating: int) -> str:
    """
    Lightroom color label の運用ルール

    Green  : 採用確定（accepted_flag=1）
    Yellow : 惜しい（★3以上だが未採用）
    Red    : 問題あり（★1以下）
    None   : その他
    """
    if accepted_flag == 1:
        return "Green"

    if rating >= 3:
        return "Yellow"

    if rating <= 1:
        return "Red"

    return ""


def shorten_reason_for_lr(reason: str, max_len: int = 90) -> str:
    """
    Lightroom のキーワード用に短文化する。
    - accepted=1 の写真だけに付与される前提
    - 冒頭に ACC: を付けて検索性を上げる
    """
    if not reason:
        return ""

    s = str(reason)

    # 表現を圧縮（LRで邪魔にならない）
    s = s.replace(" | ", " / ")
    s = s.replace("(tech=", "t=")
    s = s.replace(" face=", " f=")
    s = s.replace(" comp=", " c=")
    s = s.replace("overall=", "o=")

    # 余分な空白を削除
    s = " ".join(s.split())

    # ★ おまけ：accepted 写真の明示トークン
    s = "ACC:" + s

    return s[:max_len]


def to_labelcolor_key(color: str) -> str:
    """
    CSV内の lr_color_label ("Green" etc) を XMPの photoshop:LabelColor 用キーに寄せる。
    """
    if not color:
        return "none"
    c = str(color).strip().lower()
    # "Green" みたいなのが来ても lower で "green" になる
    if c in {"red", "yellow", "green", "blue", "purple"}:
        return c
    return "none"

def to_label_display_from_key(label_key: str) -> str:
    return LR_LABEL_DISPLAY_JA.get(label_key, "")


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

        # P95 キャリブレーション格納
        self.calibration: Dict[str, float] = {}

        # get_data の二重呼び出し対策（setupで読み込んだ結果を再利用）
        self._cached_data: Optional[List[Dict[str, Any]]] = None

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

        # ここで super().setup() を呼ぶと get_data() が走るので、
        # get_data() 側でキャッシュ/キャリブレーションを作る設計にする
        super().setup()

    def get_data(self) -> List[Dict[str, Any]]:
        # 既に読み込み済みならそれを返す（process() が再度 get_data() を呼ぶため）
        if self._cached_data is not None:
            return self._cached_data

        input_csv = (
            Path(self.paths["evaluation_data_dir"])
            / f"evaluation_results_{self.date}.csv"
        )
        if not input_csv.exists():
            raise FileNotFoundError(input_csv)

        self.logger.info(f"Loading CSV: {input_csv}")
        with input_csv.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # P95 キャリブレーションを構築
        self._build_calibration(rows)

        # キャッシュ
        self._cached_data = rows
        return rows

    def _build_calibration(self, rows: List[Dict[str, Any]]) -> None:
        """
        ②：固定 max をやめて「その日の P95」を max として使う（外れ値に強い）
        """
        def p95_of(key: str, floor: float = 1e-6) -> float:
            vals = []
            for r in rows:
                v = r.get(key)
                if v in ("", None):
                    continue
                fv = safe_float(v)
                if fv > 0:
                    vals.append(fv)
            p95v = percentile(vals, 95) if vals else floor
            return max(floor, float(p95v))

        # tech
        self.calibration = {
            "tech_sharpness": p95_of("sharpness_score"),
            "tech_local_sharpness": p95_of("local_sharpness_score"),
            "tech_noise": p95_of("noise_score"),
            # blurriness は固定
            "tech_blurriness": BLURRINESS_MAX,

            # face
            "face_sharpness": p95_of("face_sharpness_score"),
            "face_contrast": p95_of("face_contrast_score"),
            "face_noise": p95_of("face_noise_score"),
            "face_local_sharpness": p95_of("face_local_sharpness_score"),
            "face_local_contrast": p95_of("face_local_contrast_score"),
        }

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

                tech_score, tech_bd = self._technical_score(row)
                comp_score, comp_bd = self._composition_score(row)
                face_score, face_bd = self._face_score(row) if face_detected else (0.0, {})

                # overall（今の設計を維持）
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
    # scoring (with breakdown)
    # -------------------------

    def _technical_score(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        bd = {
            "sharpness": weighted_contribution(
                safe_float(row.get("sharpness_score")),
                self.calibration.get("tech_sharpness", 1.0),
                TECH_WEIGHTS["sharpness"],
            ),
            "local_sharpness": weighted_contribution(
                safe_float(row.get("local_sharpness_score")),
                self.calibration.get("tech_local_sharpness", 1.0),
                TECH_WEIGHTS["local_sharpness"],
            ),
            "noise": weighted_contribution(
                safe_float(row.get("noise_score")),
                self.calibration.get("tech_noise", 1.0),
                TECH_WEIGHTS["noise"],
                inverse=True,
            ),
            "blurriness": weighted_contribution(
                safe_float(row.get("blurriness_score")),
                self.calibration.get("tech_blurriness", BLURRINESS_MAX),
                TECH_WEIGHTS["blurriness"],
                inverse=True,
            ),
        }
        return sum(bd.values()) * 100.0, bd

    def _face_score(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        bd = {
            "sharpness": weighted_contribution(
                safe_float(row.get("face_sharpness_score")),
                self.calibration.get("face_sharpness", 1.0),
                FACE_WEIGHTS["sharpness"],
            ),
            "contrast": weighted_contribution(
                safe_float(row.get("face_contrast_score")),
                self.calibration.get("face_contrast", 1.0),
                FACE_WEIGHTS["contrast"],
            ),
            "noise": weighted_contribution(
                safe_float(row.get("face_noise_score")),
                self.calibration.get("face_noise", 1.0),
                FACE_WEIGHTS["noise"],
                inverse=True,
            ),
            "local_sharpness": weighted_contribution(
                safe_float(row.get("face_local_sharpness_score")),
                self.calibration.get("face_local_sharpness", 1.0),
                FACE_WEIGHTS["local_sharpness"],
            ),
            "local_contrast": weighted_contribution(
                safe_float(row.get("face_local_contrast_score")),
                self.calibration.get("face_local_contrast", 1.0),
                FACE_WEIGHTS["local_contrast"],
            ),
        }
        return sum(bd.values()) * 100.0, bd

    def _composition_score(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
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

        # ===== category 付与（まずは face_detected で確定）=====
        for r in rows:
            face = bool(r.get("face_detected"))
            r["category"] = "portrait" if face else "non_face"

        # ===== ① accepted_flag：カテゴリ別×分布ベース（percentile AND max_accept）=====
        portrait_rows = [r for r in rows if r["category"] == "portrait"]
        non_face_rows = [r for r in rows if r["category"] == "non_face"]

        portrait_scores = [safe_float(r.get("overall_score")) for r in portrait_rows]
        non_face_scores = [safe_float(r.get("overall_score")) for r in non_face_rows]

        portrait_thr = percentile(portrait_scores, ACCEPT_RULES["portrait"]["percentile"]) if portrait_scores else 0.0
        non_face_thr = percentile(non_face_scores, ACCEPT_RULES["non_face"]["percentile"]) if non_face_scores else 0.0

        portrait_rows.sort(key=lambda r: safe_float(r.get("overall_score")), reverse=True)
        non_face_rows.sort(key=lambda r: safe_float(r.get("overall_score")), reverse=True)

        # accepted_reason 生成に使う: contrib列を集めるヘルパ
        def extract_contrib(prefix: str, r: Dict[str, Any]) -> Dict[str, Any]:
            # prefix は "contrib_tech_" / "contrib_face_" / "contrib_comp_"
            out = {}
            for k, v in r.items():
                if k.startswith(prefix):
                    out[k[len(prefix):]] = v
            return out

        # portrait
        for i, r in enumerate(portrait_rows):
            rank = i + 1
            max_accept = ACCEPT_RULES["portrait"]["max_accept"]
            overall = safe_float(r.get("overall_score"))
            ok = (i < max_accept) and (overall >= portrait_thr)
            r["accepted_flag"] = int(ok)

            r["accepted_reason"] = build_accepted_reason(
                category="portrait",
                rank=rank,
                max_accept=max_accept,
                threshold=portrait_thr,
                overall=overall,
                accepted_flag=r["accepted_flag"],   # ← 追加
                score_technical=safe_float(r.get("score_technical")),
                score_face=safe_float(r.get("score_face")),
                score_composition=safe_float(r.get("score_composition")),
                contrib_tech=extract_contrib("contrib_tech_", r),
                contrib_face=extract_contrib("contrib_face_", r),
                contrib_comp=extract_contrib("contrib_comp_", r),
                top_n=3,  # ← 上位3要素
            )

            if r["accepted_flag"] == 0:
                r["accepted_reason"] = ""

        # non_face
        for i, r in enumerate(non_face_rows):
            rank = i + 1
            max_accept = ACCEPT_RULES["non_face"]["max_accept"]
            overall = safe_float(r.get("overall_score"))
            ok = (i < max_accept) and (overall >= non_face_thr)
            r["accepted_flag"] = int(ok)

            r["accepted_reason"] = build_accepted_reason(
                category="non_face",
                rank=rank,
                max_accept=max_accept,
                threshold=non_face_thr,
                overall=overall,
                accepted_flag=r["accepted_flag"],   # ← 追加
                score_technical=safe_float(r.get("score_technical")),
                score_face=None,
                score_composition=safe_float(r.get("score_composition")),
                contrib_tech=extract_contrib("contrib_tech_", r),
                contrib_face={},
                contrib_comp=extract_contrib("contrib_comp_", r),
                top_n=3,
            )

            if r["accepted_flag"] == 0:
                r["accepted_reason"] = ""

        # ===== flag：従来通り（全体上位比率）=====
        rows.sort(key=lambda r: safe_float(r.get("overall_score")), reverse=True)
        top_n = max(1, int(len(rows) * TOP_FLAG_RATIO))
        for i, r in enumerate(rows):
            r["flag"] = 1 if i < top_n else 0

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

        # breakdown列は「存在するものだけ」末尾に追加（漏れ防止 & 安定）
        extra_cols = set()
        for r in rows:
            for k in r.keys():
                if k.startswith("score_") or k.startswith("contrib_"):
                    extra_cols.add(k)
        extra_columns = sorted(extra_cols)

        # 最終列にしたいので最後に追加
        columns = base_columns + extra_columns + [
            "lr_keywords", "lr_rating",
            "lr_color_label",         # 既存
            "lr_labelcolor_key",      # 追加（photoshop:LabelColor用）
            "lr_label_display",       # 追加（xmp:Label用）
            "accepted_reason",
        ]

        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for r in rows:
                overall = safe_float(r.get("overall_score"))
                rating = score_to_rating(overall)

                r["lr_rating"] = rating
                label = choose_color_label(r.get("category",""), int(r.get("accepted_flag",0) or 0), rating)

                r["lr_color_label"] = label  # 既存互換として残す（任意）
                r["lr_labelcolor_key"] = to_labelcolor_key(label)              # ← 追加
                r["lr_label_display"] = to_label_display_from_key(r["lr_labelcolor_key"])  # ← 追加

                # accepted_reason -> lr_keywords（accepted_flag=0 は空）
                reason = r.get("accepted_reason", "") or ""
                if int(r.get("accepted_flag", 0) or 0) == 1:
                    r["lr_keywords"] = shorten_reason_for_lr(reason, max_len=90)
                else:
                    r["lr_keywords"] = ""

                r_out = dict(r)

                # face_detected 表記だけ互換（TRUE/FALSE）
                r_out["face_detected"] = "TRUE" if bool(r_out.get("face_detected")) else "FALSE"

                # 数値は丸め（必要ならここを調整）
                # ※ 入力由来の raw 値はそのまま残す（評価結果系は format_score 済み）
                writer.writerow({k: r_out.get(k) for k in columns})

        self.logger.info(f"Output written: {output_csv}")
        self.logger.info(
            f"Accepted thresholds: portrait(P{ACCEPT_RULES['portrait']['percentile']}={portrait_thr:.2f}), "
            f"non_face(P{ACCEPT_RULES['non_face']['percentile']}={non_face_thr:.2f})"
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
