# src/batch_processor/evaluation_rank/scoring.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

# =========================
# weights (scoring only)
# =========================

# NOTE:
# - 技術は「良いなら横並び」「弱点だけ減点」へ移行したため、TECH_WEIGHTS は互換のため残す。
# - breakdown（contrib_*）は “0..1” を返す（BatchProcessor側で *100 される前提）
TECH_WEIGHTS = {
    "sharpness": 0.30,
    "local_sharpness": 0.28,
    "noise": 0.20,
    "blurriness": 0.12,
    "exposure": 0.10,
}

# 顔スコア: 顔シャープ・顔局所シャープを厚め、表情/コントラストを次点
FACE_WEIGHTS = {
    "sharpness": 0.22,
    "local_sharpness": 0.20,
    "expression": 0.16,
    "contrast": 0.10,
    "noise": 0.10,
    "local_contrast": 0.10,
    "exposure": 0.12,
}

# 構図スコア: ルール系 + 顔位置 + フレーミングが軸。アイコンタクト/余白も効く。
COMPOSITION_WEIGHTS = {
    "composition_rule_based_score": 0.22,
    "face_position_score": 0.20,
    "framing_score": 0.18,
    "eye_contact_score": 0.14,
    "face_direction_score": 0.08,
    "lead_room_score": 0.08,
    "body_composition_score": 0.05,
    "rule_of_thirds_score": 0.05,
}

# full_body のとき「構図寄り」へ最大この割合だけシフト（互換として残す）
FULL_BODY_COMP_SHIFT_MAX = 0.12


# =========================
# status / parsing
# =========================


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v in ("", None):
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return False


def is_ok_status(v: Any) -> bool:
    """
    ok / true / 1 / success など “信頼できる” を広めに拾う。
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"ok", "true", "t", "1", "success"}  # ★ "ok" を必ず拾う


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def clamp100(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 100.0:
        return 100.0
    return x


def score01(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """
    コントラクト上 score は 0..1（離散）前提。
    念のため 0..100 が紛れたら 0..1 に戻す。
    """
    v = safe_float(row.get(key), default=default)
    if v > 1.5:
        v = v / 100.0
    return clamp01(v)


def pick_score(row: Dict[str, Any], primary_key: str, fallback_key: str) -> float:
    """
    brightness_adjusted があれば優先、なければ通常score。
    """
    if primary_key in row and row.get(primary_key) not in ("", None):
        return score01(row, primary_key, default=0.0)
    return score01(row, fallback_key, default=0.0)


def parse_gaze_y(v: Any) -> Optional[float]:
    """
    gaze は "{'x':..., 'y':..., 'z':...}" のような dict 文字列が来る想定。
    安全に y を取り出す。取れなければ None。
    """
    if v in ("", None):
        return None
    if isinstance(v, dict):
        y = v.get("y")
        try:
            return float(y)
        except (TypeError, ValueError):
            return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                y = d.get("y")
                try:
                    return float(y)
                except (TypeError, ValueError):
                    return None
        except Exception:
            return None
    return None


def half_closed_eye_penalty_proxy(row: Dict[str, Any]) -> float:
    """
    現状ランドマーク不足につき「半目そのもの」は測れない。
    代理として「強い俯き + 下向き視線 + アイコンタクト低下」を半目っぽさとしてペナルティ化。

    返り値: 0..1（大きいほど悪い）
    """
    if not safe_bool(row.get("face_detected")):
        return 0.0

    pitch = safe_float(row.get("debug_pitch", row.get("pitch", 0.0)), 0.0)  # 下向きは負方向を想定

    gaze_y_raw = row.get("debug_gaze_y", None)
    if gaze_y_raw in ("", None):
        gaze_y = parse_gaze_y(row.get("gaze"))
    else:
        try:
            gaze_y = float(gaze_y_raw)
        except (TypeError, ValueError):
            gaze_y = parse_gaze_y(row.get("gaze"))

    eye = score01(row, "debug_eye_contact", default=score01(row, "eye_contact_score", default=0.0))

    if gaze_y is None:
        if pitch <= -35.0 and eye <= 0.85:
            return 0.25
        return 0.0

    if pitch <= -45.0 or gaze_y <= -0.60:
        return 0.60

    if pitch <= -25.0 and gaze_y <= -0.35 and eye <= 0.85:
        return 0.35

    if pitch <= -30.0 and (gaze_y <= -0.30 or eye <= 0.80):
        return 0.15

    return 0.0


def apply_half_closed_penalty_to_expression(expr01: float, penalty01: float) -> float:
    """
    expression の寄与にだけ“冷たく”効かせる。
    - penalty=0.60 でも expression をゼロにせず、強めに下げる程度に留める。
    """
    expr01 = clamp01(expr01)
    penalty01 = clamp01(penalty01)

    # multiplier: 1.0 .. 0.55（強でも45%減まで）
    mult = 1.0 - 0.75 * penalty01
    if mult < 0.55:
        mult = 0.55
    return expr01 * mult


# =========================
# scorer
# =========================


@dataclass
class ScorePack:
    score: float
    breakdown: Dict[str, float]  # 0..1（Batch側で *100）


@dataclass
class EvaluationScorer:
    """
    新INPUT（score/raw/status分離）前提のスコアラー。
    - score は 0..100 固定
    - breakdown は 0..1 固定（BatchProcessor で *100 して CSV に出る）
    - tech の breakdown は “弱点ペナルティ量” を正値で返す（0..1）
    """

    # status が怪しいときの “減点縮退” 係数（=悪いと断定しない）
    RELIABILITY_DECAY_DEFAULT: float = 0.65
    RELIABILITY_DECAY_SHARPNESS: float = 0.45  # ピント系は厳しめ
    RELIABILITY_DECAY_BLUR: float = 0.55  # ブレ系も厳しめ

    def _reliability(self, ok: bool, *, kind: str) -> float:
        if ok:
            return 1.0
        if kind == "sharpness":
            return float(self.RELIABILITY_DECAY_SHARPNESS)
        if kind == "blurriness":
            return float(self.RELIABILITY_DECAY_BLUR)
        return float(self.RELIABILITY_DECAY_DEFAULT)

    def _shot_type(self, row: Dict[str, Any]) -> str:
        s = str(row.get("shot_type") or "").strip().lower()
        return s or "unknown"

    # -------------------------
    # technical (penalty-first)
    # -------------------------
    def _tech_penalty_points(
        self,
        sharp: float,
        local_sharp: float,
        noise: float,
        blur: float,
        expo: float,
    ) -> Dict[str, float]:
        """
        技術は「良いなら横並び」「弱点だけ減点」。
        返り値:
          penalty_bd_points: 軸別ペナルティ（0..100 の “点” / 正値）
        """
        bd: Dict[str, float] = {
            "sharpness": 0.0,
            "local_sharpness": 0.0,
            "noise": 0.0,
            "blurriness": 0.0,
            "exposure": 0.0,
        }

        # 破綻（足切りに近い減点）
        if sharp < 0.50:
            bd["sharpness"] += 18.0
        if local_sharp < 0.50:
            bd["local_sharpness"] += 16.0
        if noise < 0.50:
            bd["noise"] += 12.0
        if blur < 0.50:
            bd["blurriness"] += 12.0
        if expo < 0.40:
            bd["exposure"] += 10.0

        # “惜しい” 減点（0.5〜0.75の弱点）
        if 0.50 <= sharp < 0.75:
            bd["sharpness"] += 6.0
        if 0.50 <= local_sharp < 0.75:
            bd["local_sharpness"] += 5.0
        if 0.50 <= noise < 0.75:
            bd["noise"] += 4.0
        if 0.50 <= blur < 0.75:
            bd["blurriness"] += 4.0
        if 0.40 <= expo < 0.75:
            bd["exposure"] += 3.0

        # 念のため（負値は絶対出さない）
        for k in list(bd.keys()):
            if bd[k] < 0.0:
                bd[k] = 0.0
        return bd

    def technical_score(self, row: Dict[str, Any]) -> ScorePack:
        sharp = score01(row, "sharpness_score")
        local_sharp = score01(row, "local_sharpness_score")
        noise = pick_score(row, "noise_score_brightness_adjusted", "noise_score")
        blur = pick_score(row, "blurriness_score_brightness_adjusted", "blurriness_score")
        expo = score01(row, "exposure_score")

        # 信頼度（status）
        sharp_ok = is_ok_status(row.get("sharpness_eval_status"))
        local_ok = is_ok_status(row.get("local_sharpness_eval_status"))
        noise_ok = is_ok_status(row.get("noise_eval_status"))
        blur_ok = is_ok_status(row.get("blurriness_eval_status"))
        expo_ok = is_ok_status(row.get("exposure_eval_status"))

        base = 100.0

        penalty_bd_points = self._tech_penalty_points(sharp, local_sharp, noise, blur, expo)

        # “不確実”なときはペナルティを縮退（=悪いと断定しない）
        rel = 1.0
        rel *= 1.0 if sharp_ok else self.RELIABILITY_DECAY_SHARPNESS
        rel *= 1.0 if local_ok else self.RELIABILITY_DECAY_SHARPNESS
        rel *= 1.0 if blur_ok else self.RELIABILITY_DECAY_BLUR
        rel *= 1.0 if noise_ok else self.RELIABILITY_DECAY_DEFAULT
        rel *= 1.0 if expo_ok else self.RELIABILITY_DECAY_DEFAULT
        if rel < 0.55:
            rel = 0.55

        # 縮退適用
        penalty_bd_points = {k: float(v) * float(rel) for k, v in penalty_bd_points.items()}
        penalty_total = sum(penalty_bd_points.values())

        score = base - penalty_total
        score = clamp100(score)  # ★ 0..100 固定（0未満禁止）

        # breakdown は “0..1 のペナルティ量（正値）”
        bd01 = {k: clamp01(float(v) / 100.0) for k, v in penalty_bd_points.items()}

        return ScorePack(score=float(score), breakdown=bd01)

    def _gate_penalty_tech(self, row: Dict[str, Any]) -> float:
        """
        互換用（旧実装呼び出しの名残）。
        技術は penalty-first に移行したため、ここは常に 1.0 を返す。
        """
        return 1.0

    # -------------------------
    # face
    # -------------------------
    def face_score(self, row: Dict[str, Any]) -> ScorePack:
        if not safe_bool(row.get("face_detected")):
            return ScorePack(score=0.0, breakdown={})

        fsharp = score01(row, "face_sharpness_score")
        flocal = score01(row, "face_local_sharpness_score")
        fexpo = score01(row, "face_exposure_score")
        fnoise = score01(row, "face_noise_score")
        fcont = score01(row, "face_contrast_score")
        flocal_cont = score01(row, "face_local_contrast_score")
        expr = score01(row, "expression_score")

        half_pen = half_closed_eye_penalty_proxy(row)
        expr_effective = apply_half_closed_penalty_to_expression(expr, half_pen)

        fsharp_r = self._reliability(is_ok_status(row.get("face_sharpness_eval_status")), kind="sharpness")
        flocal_r = 1.0  # eval_status列が無い設計なら 1.0
        fexpo_r = self._reliability(is_ok_status(row.get("face_exposure_eval_status")), kind="exposure")
        fcont_r = self._reliability(is_ok_status(row.get("face_contrast_eval_status")), kind="contrast")
        fnoise_r = 1.0  # face_noise_eval_status が無い設計なら 1.0

        bd = {
            "sharpness": fsharp * FACE_WEIGHTS["sharpness"] * fsharp_r,
            "local_sharpness": flocal * FACE_WEIGHTS["local_sharpness"] * flocal_r,
            "expression": expr_effective * FACE_WEIGHTS["expression"] * 1.0,
            "contrast": fcont * FACE_WEIGHTS["contrast"] * fcont_r,
            "noise": fnoise * FACE_WEIGHTS["noise"] * fnoise_r,
            "local_contrast": flocal_cont * FACE_WEIGHTS["local_contrast"] * 1.0,
            "exposure": fexpo * FACE_WEIGHTS["exposure"] * fexpo_r,
        }

        score = sum(bd.values()) * 100.0
        score *= self._pose_penalty(row)
        score *= self._expression_penalty(expr)
        score = clamp100(score)

        # breakdown は 0..1 期待なので clamp
        bd01 = {k: clamp01(float(v)) for k, v in bd.items()}
        return ScorePack(score=float(score), breakdown=bd01)

    def _pose_penalty(self, row: Dict[str, Any]) -> float:
        yaw = abs(safe_float(row.get("yaw", 0.0)))
        pitch = abs(safe_float(row.get("pitch", 0.0)))

        yaw_factor = 1.0 - clamp01(yaw / 45.0) * 0.25  # 最大 -25%
        pitch_factor = 1.0 - clamp01(pitch / 45.0) * 0.25
        return yaw_factor * pitch_factor

    def _expression_penalty(self, expr_score01: float) -> float:
        if expr_score01 >= 0.75:
            return 1.00
        if expr_score01 >= 0.50:
            return 0.92
        if expr_score01 >= 0.25:
            return 0.80
        return 0.68

    # -------------------------
    # composition (shot-type aware)
    # -------------------------
    def _composition_weights_for_shot(self, row: Dict[str, Any]) -> Dict[str, float]:
        st = self._shot_type(row)
        w = dict(COMPOSITION_WEIGHTS)

        if st in {"full_body", "seated"}:
            w["eye_contact_score"] *= 0.65
            w["face_direction_score"] *= 0.75

            w["framing_score"] *= 1.18
            w["body_composition_score"] *= 1.80
            w["lead_room_score"] *= 1.15

            w["composition_rule_based_score"] *= 1.10
            w["rule_of_thirds_score"] *= 1.10

        elif st == "upper_body":
            w["framing_score"] *= 1.08
            w["composition_rule_based_score"] *= 1.05
            w["eye_contact_score"] *= 0.90

        total = sum(w.values())
        if total > 0:
            for k in list(w.keys()):
                w[k] = w[k] / total
        return w

    def composition_score(self, row: Dict[str, Any]) -> ScorePack:
        st = self._shot_type(row)
        full_body = safe_bool(row.get("full_body_detected"))
        bd: Dict[str, float] = {}

        wmap = self._composition_weights_for_shot(row)

        for k, w in wmap.items():
            bd[k] = score01(row, k) * w

        score = sum(bd.values()) * 100.0

        if st in {"full_body", "seated"} or full_body:
            score *= 1.02

        # -------------------------
        # reliability gate (contract-first)
        # -------------------------
        # 新契約: composition_eval_status = ok / fallback_used / invalid
        ev = str(row.get("composition_eval_status") or "").strip().lower()

        if ev == "ok":
            score *= 1.0
        elif ev == "fallback_used":
            # 「分からないので0.5」系は、悪いと断定しないが少しだけ縮退
            score *= 0.85
        elif ev == "invalid":
            score *= 0.75
        else:
            # 互換: 古いCSVは composition_status しか無いので、状態文字列として扱う
            # ok相当: face_only/body_only/face_and_body/... は “計算できた” なので 1.0
            cs = str(row.get("composition_status") or "").strip().lower()
            if cs in {
                "face_only",
                "body_only",
                "face_and_body",
                "rule_of_thirds_only",
            } or ("_and_" in cs):
                score *= 1.0
            elif cs in {"not_computed", "not_computed_with_default"}:
                score *= 0.85
            elif cs in {"invalid"}:
                score *= 0.75
            else:
                # 何も分からない場合はニュートラルに軽く縮退
                score *= 0.90

        score = clamp100(score)
        bd01 = {k: clamp01(float(v)) for k, v in bd.items()}
        return ScorePack(score=float(score), breakdown=bd01)

    # -------------------------
    # shot-type resilience bonus
    # -------------------------
    def _shot_type_resilience_bonus(self, row: Dict[str, Any]) -> float:
        st = self._shot_type(row)
        if st not in {"full_body", "seated", "upper_body"}:
            return 0.0

        pose = safe_float(row.get("pose_score"), 0.0)
        if pose > 1.5:
            pose = pose / 100.0
        pose = clamp01(pose)

        cut = safe_float(row.get("full_body_cut_risk"), 0.0)
        if cut > 1.5:
            cut = cut / 100.0
        cut = clamp01(cut)

        quality = pose * (1.0 - cut)

        if st == "upper_body":
            quality = max(pose, quality * 0.85)

        # ボーナスは overall の “点” として加える（最大+2点）
        return 2.0 * clamp01(quality)

    # -------------------------
    # overall
    # -------------------------
    def overall_score(
        self,
        row: Dict[str, Any],
        tech: ScorePack,
        face: ScorePack,
        comp: ScorePack,
    ) -> float:
        face_detected = safe_bool(row.get("face_detected"))
        full_body = safe_bool(row.get("full_body_detected"))
        st = self._shot_type(row)

        if face_detected:
            if st in {"full_body", "seated"} or full_body:
                w_face, w_tech, w_comp = 0.40, 0.15, 0.45
            elif st == "upper_body":
                w_face, w_tech, w_comp = 0.50, 0.15, 0.35
            else:
                w_face, w_tech, w_comp = 0.55, 0.15, 0.30
        else:
            w_face, w_tech, w_comp = 0.00, 0.25, 0.75

        if full_body:
            pose = safe_float(row.get("pose_score"), 0.0)
            if pose > 1.5:
                pose = pose / 100.0
            pose = clamp01(pose)

            cut = safe_float(row.get("full_body_cut_risk"), 0.0)
            if cut > 1.5:
                cut = cut / 100.0
            cut = clamp01(cut)

            fb_quality = pose * (1.0 - cut)
            shift = fb_quality * FULL_BODY_COMP_SHIFT_MAX

            if face_detected:
                w_comp += shift
                w_face -= shift * 0.60
                w_tech -= shift * 0.40
            else:
                w_comp += shift
                w_tech -= shift

            total = w_face + w_tech + w_comp
            if total > 0:
                w_face /= total
                w_tech /= total
                w_comp /= total

        base = (w_face * float(face.score)) + (w_tech * float(tech.score)) + (w_comp * float(comp.score))
        bonus = self._shot_type_resilience_bonus(row)

        out = base + bonus
        return float(clamp100(out))

    # -------------------------
    # backward compat (old pipeline)
    # -------------------------
    def build_calibration(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        return {}

    @property
    def calibration(self) -> dict[str, float]:
        return {}

    def technical_score_compat(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        p = self.technical_score(row)
        return p.score, p.breakdown

    def face_score_compat(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        p = self.face_score(row)
        return p.score, p.breakdown

    def composition_score_compat(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        p = self.composition_score(row)
        return p.score, p.breakdown
