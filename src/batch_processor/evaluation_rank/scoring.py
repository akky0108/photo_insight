# src/batch_processor/evaluation_rank/scoring.py
# -*- coding: utf-8 -*-
from __future__ import annotations


import ast

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional


# =========================
# weights (scoring only)
# =========================

# 技術スコア: ピント&局所ピントを最重視、次にノイズ、次にブレ、最後に露出
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

# full_body のとき「構図寄り」へ最大この割合だけシフト
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
    # 最後の保険
    try:
        return bool(int(float(s)))
    except Exception:
        return False


def is_ok_status(v: Any) -> bool:
    """
    ok / true / 1 / SUCCESS など “信頼できる” を広めに拾う。
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in {"ok", "true", "t", "1", "success"}


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
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

    pitch = safe_float(row.get("pitch"), 0.0)  # 下向きは負方向を想定
    gaze_y = parse_gaze_y(row.get("gaze"))
    eye = score01(row, "eye_contact_score", default=0.0)

    # gaze が取れないときは「俯き + アイコンタクト低下」だけで弱く反応
    if gaze_y is None:
        if pitch <= -35.0 and eye <= 0.85:
            return 0.25
        return 0.0

    # 強: かなり俯いてる or かなり下目線
    if pitch <= -45.0 or gaze_y <= -0.60:
        return 0.60

    # 中: 俯き + 下目線 + アイコンタクト低め（半目が出やすい条件）
    if pitch <= -25.0 and gaze_y <= -0.35 and eye <= 0.85:
        return 0.35

    # 弱: どれか一つだけ強い（軽く下げる）
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
    breakdown: Dict[str, float]


@dataclass
class EvaluationScorer:
    """
    新INPUT（score/raw/status分離）前提のスコアラー。
    - scoreは0..1主指標
    - rawは理由/タイブレーク用（基本スコア計算の主軸にはしない）
    - statusがokでない場合は “その軸の寄与を縮退”
    """

    # status が怪しいときの寄与縮退係数（“悪い”ではなく“不確実”）
    RELIABILITY_DECAY_DEFAULT: float = 0.65
    RELIABILITY_DECAY_SHARPNESS: float = 0.45  # ピント系は厳しめ
    RELIABILITY_DECAY_BLUR: float = 0.55       # ブレ系も厳しめ

    def _reliability(self, ok: bool, *, kind: str) -> float:
        if ok:
            return 1.0
        if kind == "sharpness":
            return self.RELIABILITY_DECAY_SHARPNESS
        if kind == "blurriness":
            return self.RELIABILITY_DECAY_BLUR
        return self.RELIABILITY_DECAY_DEFAULT

    # -------------------------
    # technical
    # -------------------------
    def technical_score(self, row: Dict[str, Any]) -> ScorePack:
        sharp = score01(row, "sharpness_score")
        local_sharp = score01(row, "local_sharpness_score")
        noise = pick_score(row, "noise_score_brightness_adjusted", "noise_score")
        blur = pick_score(row, "blurriness_score_brightness_adjusted", "blurriness_score")
        expo = score01(row, "exposure_score")

        # 信頼度（status）
        sharp_r = self._reliability(is_ok_status(row.get("sharpness_eval_status")), kind="sharpness")
        local_r = self._reliability(is_ok_status(row.get("local_sharpness_eval_status")), kind="sharpness")
        noise_r = self._reliability(is_ok_status(row.get("noise_eval_status")), kind="noise")
        blur_r = self._reliability(is_ok_status(row.get("blurriness_eval_status")), kind="blurriness")
        expo_r = self._reliability(is_ok_status(row.get("exposure_eval_status")), kind="exposure")

        bd = {
            "sharpness": sharp * TECH_WEIGHTS["sharpness"] * sharp_r,
            "local_sharpness": local_sharp * TECH_WEIGHTS["local_sharpness"] * local_r,
            "noise": noise * TECH_WEIGHTS["noise"] * noise_r,
            "blurriness": blur * TECH_WEIGHTS["blurriness"] * blur_r,
            "exposure": expo * TECH_WEIGHTS["exposure"] * expo_r,
        }

        # 0..1 → 0..100 に
        score = sum(bd.values()) * 100.0
        score *= self._gate_penalty_tech(row)
        return ScorePack(score=score, breakdown=bd)

    def _gate_penalty_tech(self, row: Dict[str, Any]) -> float:
        """
        “破綻”を落とすゲート（ただし落とし過ぎない）
        - ブレ/露出が極端に悪いなら軽く減衰
        """
        blur = pick_score(row, "blurriness_score_brightness_adjusted", "blurriness_score")
        expo = score01(row, "exposure_score")

        # blur 0.25以下はだいぶ厳しい、0.5以上はOK、1.0は満点
        blur_factor = 0.75 + 0.25 * clamp01((blur - 0.25) / (0.50 - 0.25))
        # expo 0.25以下は厳しい、0.5以上はOK
        expo_factor = 0.80 + 0.20 * clamp01((expo - 0.25) / (0.50 - 0.25))
        return blur_factor * expo_factor

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

        # 半目っぽさ（代理）を expression にだけ反映（“冷たい”）
        half_pen = half_closed_eye_penalty_proxy(row)
        expr_effective = apply_half_closed_penalty_to_expression(expr, half_pen)

        fsharp_r = self._reliability(is_ok_status(row.get("face_sharpness_eval_status")), kind="sharpness")
        # face_local_sharpness は “eval_status列が無い” 前提で保守的に扱う（あれば追加してOK）
        flocal_r = 1.0
        fexpo_r = self._reliability(is_ok_status(row.get("face_exposure_eval_status")), kind="exposure")
        fcont_r = self._reliability(is_ok_status(row.get("face_contrast_eval_status")), kind="contrast")
        fnoise_r = 1.0  # face_noise_eval_statusが無ければ1.0（あれば noise_eval_status 的に追加）

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
        return ScorePack(score=score, breakdown=bd)

    def _pose_penalty(self, row: Dict[str, Any]) -> float:
        """
        yaw/pitch が大きい場合に軽く減衰（極端な横顔や俯きで “採用感” を下げる）
        """
        yaw = abs(safe_float(row.get("yaw"), 0.0))
        pitch = abs(safe_float(row.get("pitch"), 0.0))

        # 0..45度は緩やか、45以上で頭打ち
        yaw_factor = 1.0 - clamp01(yaw / 45.0) * 0.25   # 最大 -25%
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
    # composition
    # -------------------------
    def composition_score(self, row: Dict[str, Any]) -> ScorePack:
        full_body = safe_bool(row.get("full_body_detected"))
        bd: Dict[str, float] = {}

        # すべて score 0..1 前提
        for k, w in COMPOSITION_WEIGHTS.items():
            bd[k] = score01(row, k) * w

        score = sum(bd.values()) * 100.0

        # 全身は “構図の意味” が増える（ただしやり過ぎない）
        if full_body:
            score *= 1.02  # ほんの少しだけ押し上げ

        # composition_status が怪しければ縮退（あくまで不確実）
        comp_ok = is_ok_status(row.get("composition_status"))
        score *= (1.0 if comp_ok else 0.75)

        return ScorePack(score=score, breakdown=bd)

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

        # ベース重み
        if face_detected:
            w_face, w_tech, w_comp = 0.50, 0.35, 0.15
        else:
            w_face, w_tech, w_comp = 0.00, 0.65, 0.35

        # full_body が “良い全身” なら構図寄りへ少しシフト
        if full_body:
            pose = safe_float(row.get("pose_score"), 0.0)
            if pose > 1.5:
                pose = pose / 100.0
            pose = clamp01(pose)

            cut = safe_float(row.get("full_body_cut_risk"), 0.0)
            if cut > 1.5:
                cut = cut / 100.0
            cut = clamp01(cut)

            fb_quality = pose * (1.0 - cut)  # 0..1
            shift = fb_quality * FULL_BODY_COMP_SHIFT_MAX

            # tech から comp へ移す（顔ありは faceからも少し）
            if face_detected:
                w_comp += shift
                w_face -= shift * 0.60
                w_tech -= shift * 0.40
            else:
                w_comp += shift
                w_tech -= shift

            # 正規化
            total = w_face + w_tech + w_comp
            if total > 0:
                w_face /= total
                w_tech /= total
                w_comp /= total

        return (w_face * face.score) + (w_tech * tech.score) + (w_comp * comp.score)


    # -------------------------
    # backward compat (old pipeline)
    # -------------------------
    def build_calibration(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        """
        旧実装互換用。
        以前はP95等でキャリブレーションしていたが、
        現行コントラクトでは score(0..1) を主指標として扱うため不要。
        呼び出し側が残っていても落ちないようにする。
        """
        # 将来、rawベースのタイブレーク補正などを入れるならここに置ける
        return {}

    @property
    def calibration(self) -> dict[str, float]:
        """
        旧コードが scorer.calibration を参照しても落ちないようにする。
        """
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
