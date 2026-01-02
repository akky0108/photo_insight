# src/batch_processor/evaluation_rank/scoring.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional


# =========================
# parameters (scoring only)
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
            "tech_noise": p95_of("noise_score"),
            "tech_blurriness": BLURRINESS_MAX,  # blurriness は固定

            # face
            "face_sharpness": p95_of("face_sharpness_score"),
            "face_contrast": p95_of("face_contrast_score"),
            "face_noise": p95_of("face_noise_score"),
            "face_local_sharpness": p95_of("face_local_sharpness_score"),
            "face_local_contrast": p95_of("face_local_contrast_score"),
        }
        return self.calibration

    def technical_score(self, row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
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
        }
        return sum(bd.values()) * 100.0, bd

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
    ) -> float:
        # overall（今の設計を維持）
        if face_detected:
            return 0.45 * face_score + 0.35 * tech_score + 0.20 * comp_score
        return 0.55 * tech_score + 0.45 * comp_score
