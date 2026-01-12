# src/batch_processor/evaluation_rank/scoring.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional


# =========================
# parameters (scoring only)
# =========================

BLURRINESS_MAX = 1.0

# 全身のときに「構図寄り」に重みを少しシフトする最大量
FULL_BODY_COMPOSITION_BOOST = 0.15  # 最大で comp に +0.15 加算

TECH_WEIGHTS = {
    "sharpness": 0.28,
    "local_sharpness": 0.27,
    "noise": 0.22,
    "blurriness": 0.13,
    "exposure": 0.10,
}

# ★ face の local 系の影響を少し弱める
FACE_WEIGHTS = {
    "sharpness": 0.23,
    "contrast": 0.10,
    "noise": 0.10,
    "local_sharpness": 0.18,
    "local_contrast": 0.12,
    "expression": 0.17,
    "exposure": 0.10,
}

COMPOSITION_WEIGHTS = {
    "composition_rule_based_score": 0.25,
    "face_position_score": 0.20,
    "framing_score": 0.20,
    "face_direction_score": 0.10,
    "eye_contact_score": 0.10,
    "lead_room_score": 0.10,        
    "body_composition_score": 0.05, 
}


# =========================
# utilities (scoring only)
# =========================

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


def weighted_contribution(value: float, max_value: float, weight: float, inverse: bool = False) -> float:
    norm = normalize_inverse(value, max_value) if inverse else normalize(value, max_value)
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


# =========================
# scorer
# =========================

@dataclass
class EvaluationScorer:
    """
    - 日次キャリブレーション(P95)を保持
    - tech/face/comp の score と breakdown を返す
    """
    calibration: Dict[str, float] = field(default_factory=dict)

    def build_calibration(self, rows: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        固定 max をやめて「その日の P95」を max として使う（外れ値に強い）
        """
        def p95_of(key: str, floor: float = 1e-6) -> float:
            vals: List[float] = []
            for r in rows:
                v = r.get(key)
                if v in ("", None):
                    continue
                fv = safe_float(v)
                if fv > 0:
                    vals.append(fv)
            p95v = percentile(vals, 95) if vals else floor
            return max(floor, float(p95v))

        self.calibration = {
            # tech
            "tech_sharpness": p95_of("sharpness_score"),
            "tech_local_sharpness": p95_of("local_sharpness_score"),
            "tech_noise": p95_of("noise_score_brightness_adjusted"),
            "tech_blurriness": BLURRINESS_MAX,
            "tech_exposure": p95_of("exposure_score"),

            # face
            "face_sharpness": p95_of("face_sharpness_score"),
            "face_contrast": p95_of("face_contrast_score"),
            "face_noise": p95_of("face_noise_score"),
            "face_local_sharpness": p95_of("face_local_sharpness_score"),
            "face_local_contrast": p95_of("face_local_contrast_score"),
            "face_exposure": p95_of("face_exposure_score"),
        }
        return self.calibration

    def technical_score(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        # 明るさに応じて補正されたスコアがあればそちらを優先
        blur_source = (
            row.get("blurriness_score_brightness_adjusted",
                    row.get("blurriness_score"))
        )
        noise_source = (
            row.get("noise_score_brightness_adjusted",
                    row.get("noise_score"))
        )

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
                safe_float(noise_source),
                self.calibration.get("tech_noise", 1.0),
                TECH_WEIGHTS["noise"],
                inverse=True,
            ),
            "blurriness": weighted_contribution(
                safe_float(blur_source),
                self.calibration.get("tech_blurriness", BLURRINESS_MAX),
                TECH_WEIGHTS["blurriness"],
                inverse=True,
            ),
            "exposure": weighted_contribution(
                safe_float(row.get("exposure_score")),
                self.calibration.get("tech_exposure", 1.0),
                TECH_WEIGHTS["exposure"],
            ),
        }
        # bd の値は 0..weight の範囲なので、合計を 0..100 にスケーリング
        return sum(bd.values()) * 100.0, bd

    def _pose_quality(self, row: Dict[str, Any]) -> float:
        """
        yaw / pitch が大きく傾いているものを少しだけ減点する補正。
        - 0〜45度まではなだらかに減衰
        """
        yaw = abs(safe_float(row.get("yaw")))
        pitch = abs(safe_float(row.get("pitch")))

        # 45度で 0 になるような線形減衰
        yaw_factor = max(0.0, 1.0 - yaw / 45.0)
        pitch_factor = max(0.0, 1.0 - pitch / 45.0)

        # yaw / pitch を同程度重視
        return max(0.0, min((yaw_factor + pitch_factor) / 2.0, 1.0))

    def face_score(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
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
            # 表情（0〜1 のスコア前提なので max_value=1.0）
            "expression": weighted_contribution(
                safe_float(row.get("expression_score")),
                1.0,
                FACE_WEIGHTS["expression"],
            ),
            # ★ 顔の露出も評価
            "exposure": weighted_contribution(
                safe_float(row.get("face_exposure_score")),
                self.calibration.get("face_exposure", 1.0),
                FACE_WEIGHTS["exposure"],
            ),
        }

        base_score = sum(bd.values()) * 100.0
        pose_q = self._pose_quality(row)

        # ポーズが極端に悪くなければ 0.8〜1.0 くらいに収まるイメージ
        final_score = base_score * (0.8 + 0.2 * pose_q)

        return final_score, bd

    def composition_score(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        bd = {
            k: weighted_contribution(
                safe_float(row.get(k)),
                1.0,
                COMPOSITION_WEIGHTS[k],
            )
            for k in COMPOSITION_WEIGHTS
        }
        return sum(bd.values()) * 100.0, bd

    def overall_score(
        self,
        *,
        face_detected: bool,
        tech_score: float,
        face_score: float,
        comp_score: float,
        full_body_detected: bool = False,
        pose_score: Optional[float] = None,
        full_body_cut_risk: Optional[float] = None,
    ) -> float:
        """
        overall_score を計算する。
        - 通常は (face / tech / comp) の固定比率
        - full_body_detected=True のときは、
          pose_score / full_body_cut_risk に応じて「構図寄り」に重みをシフトする
        """

        # ベースの重み（従来設計）
        if face_detected:
            w_face, w_tech, w_comp = 0.42, 0.38, 0.20
        else:
            # 顔が無い場合は tech / comp のみ
            w_face, w_tech, w_comp = 0.0, 0.55, 0.45

        # ---- full body 情報がある場合は weight を調整 ----
        if full_body_detected:
            # pose_score は 0..1 / 0..100 どちらでも来てもよいように正規化
            pose = 0.0 if pose_score is None else float(pose_score)
            if pose > 1.0:
                pose /= 100.0
            pose = max(0.0, min(pose, 1.0))

            # full_body_cut_risk も 0..1 / 0..100 を想定して正規化
            cut = 0.0 if full_body_cut_risk is None else float(full_body_cut_risk)
            if cut > 1.0:
                cut /= 100.0
            cut = max(0.0, min(cut, 1.0))

            # 良い全身: ポーズ良い & 切れてない
            fb_quality = pose * (1.0 - cut)  # 0..1

            if fb_quality > 0.0:
                # fb_quality=1 のとき comp に最大 FULL_BODY_COMPOSITION_BOOST 足す
                boost = fb_quality * FULL_BODY_COMPOSITION_BOOST

                if face_detected:
                    # 顔あり全身: face / tech から構図へシフト（2:1で削る）
                    shift_face = boost * (2.0 / 3.0)
                    shift_tech = boost * (1.0 / 3.0)

                    w_face = max(0.0, w_face - shift_face)
                    w_tech = max(0.0, w_tech - shift_tech)
                    w_comp = min(1.0, w_comp + boost)
                else:
                    # 顔なし全身: tech から構図へシフト
                    shift_tech = boost
                    w_tech = max(0.0, w_tech - shift_tech)
                    w_comp = min(1.0, w_comp + boost)

                # 念のため正規化（sum=1 に戻す）
                total = w_face + w_tech + w_comp
                if total > 0:
                    w_face /= total
                    w_tech /= total
                    w_comp /= total

        return w_face * face_score + w_tech * tech_score + w_comp * comp_score