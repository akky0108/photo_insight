# src/evaluators/portrait_quality/metric_mapping.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

from photo_insight.evaluators.common.grade_contract import (
    normalize_eval_status,
    normalize_grade,
    score_to_grade,
)


@dataclass(frozen=True)
class MetricResultMapper:
    """
    evaluator の生結果(result dict)を、CSV用の標準キーへマッピングする共通化レイヤ。
    prefix="" -> global
    prefix="face_" -> face region

    - eval_status を contract に正規化（invalid_input -> invalid 等）
    - grade が無い場合は score から補完
    - grade がある場合も contract に正規化（揺れ吸収）
    """

    def _put_if_exists(
        self,
        out: Dict[str, Any],
        prefix: str,
        r: Dict[str, Any],
        keys: Iterable[str],
    ) -> None:
        for k in keys:
            if k in r:
                out[f"{prefix}{k}"] = r.get(k)

    def _normalize_status_in_out(
        self, out: Dict[str, Any], prefix: str, metric: str
    ) -> None:
        """
        out に入った {metric}_eval_status を contract に正規化する。
        無ければ何もしない（evaluate側が返してないケースもある）
        """
        k = f"{prefix}{metric}_eval_status"
        if k in out:
            out[k] = normalize_eval_status(out.get(k))

    def _normalize_grade_in_out(
        self, out: Dict[str, Any], prefix: str, metric: str
    ) -> None:
        """
        out に入った {metric}_grade を contract に正規化する。
        無ければ何もしない（score補完の前提）
        """
        k = f"{prefix}{metric}_grade"
        if k in out:
            out[k] = normalize_grade(out.get(k))

    def _ensure_grade(self, out: Dict[str, Any], prefix: str, metric: str) -> None:
        """
        grade が無い/正規化で None になった場合のみ、score から補完する。
        """
        score_k = f"{prefix}{metric}_score"
        grade_k = f"{prefix}{metric}_grade"

        # まず既存gradeを正規化（あれば）
        if grade_k in out:
            out[grade_k] = normalize_grade(out.get(grade_k))

        # 無い or 不明値で None になった場合は score から補完
        if (grade_k not in out or out.get(grade_k) is None) and score_k in out:
            out[grade_k] = score_to_grade(out.get(score_k))

    def _finalize_metric(
        self, out: Dict[str, Any], prefix: str, metric: str, *, ensure_grade: bool
    ) -> None:
        """
        共通後処理：status 正規化 + (必要なら) grade 正規化/補完
        """
        self._normalize_status_in_out(out, prefix, metric)
        if ensure_grade:
            self._ensure_grade(out, prefix, metric)

    def map(
        self, name: str, result: Dict[str, Any], prefix: str = ""
    ) -> Dict[str, Any]:
        r = result or {}
        out: Dict[str, Any] = {}

        # -------------------------
        # Noise（特殊：既存互換を維持）
        # -------------------------
        if name == "noise":
            metric = "noise"

            if prefix:
                out[f"{prefix}noise_score"] = r.get("noise_score", 0.5)
                self._put_if_exists(
                    out,
                    prefix,
                    r,
                    (
                        "noise_raw",
                        "noise_grade",
                        "noise_sigma_midtone",
                        "noise_sigma_used",
                        "noise_mask_ratio",
                        "noise_eval_status",
                        "noise_fallback_reason",
                        "noise_score_brightness_adjusted",
                    ),
                )

                # 保険：NoiseEvaluator が noise_raw を返してない場合だけ補完
                if f"{prefix}noise_raw" not in out:
                    sigma_used = r.get("noise_sigma_used")
                    try:
                        out[f"{prefix}noise_raw"] = (
                            -float(sigma_used) if sigma_used is not None else None
                        )
                    except (TypeError, ValueError):
                        out[f"{prefix}noise_raw"] = None

            else:
                out.update(r)

                # 保険：noise_raw が無い場合だけ補完
                if "noise_raw" not in out:
                    sigma_used = r.get("noise_sigma_used")
                    try:
                        out["noise_raw"] = (
                            -float(sigma_used) if sigma_used is not None else None
                        )
                    except (TypeError, ValueError):
                        out["noise_raw"] = None

            # Step B: status/grade 正規化（noise_grade が無ければ score から補完）
            self._finalize_metric(out, prefix, metric, ensure_grade=True)
            return out

        # -------------------------
        # Sharpness
        # -------------------------
        if name == "sharpness":
            metric = "sharpness"
            out[f"{prefix}sharpness_score"] = r.get("sharpness_score", 0.5)
            self._put_if_exists(
                out,
                prefix,
                r,
                (
                    "sharpness_raw",
                    "sharpness_eval_status",
                    "sharpness_fallback_reason",
                    "sharpness_grade",
                ),
            )
            self._finalize_metric(out, prefix, metric, ensure_grade=True)
            return out

        # -------------------------
        # Contrast
        # -------------------------
        if name == "contrast":
            metric = "contrast"
            out[f"{prefix}contrast_score"] = r.get("contrast_score", 0.5)
            self._put_if_exists(
                out,
                prefix,
                r,
                (
                    "contrast_raw",
                    "contrast_grade",
                    "contrast_eval_status",
                    "contrast_fallback_reason",
                ),
            )
            self._finalize_metric(out, prefix, metric, ensure_grade=True)
            return out

        # -------------------------
        # Blurriness
        # -------------------------
        if name == "blurriness":
            metric = "blurriness"
            out[f"{prefix}blurriness_score"] = r.get("blurriness_score", 0.5)
            self._put_if_exists(
                out,
                prefix,
                r,
                (
                    "blurriness_raw",
                    "blurriness_grade",
                    "blurriness_eval_status",
                    "blurriness_fallback_reason",
                    "blurriness_score_brightness_adjusted",
                ),
            )
            self._finalize_metric(out, prefix, metric, ensure_grade=True)
            return out

        # -------------------------
        # Exposure
        # -------------------------
        if name == "exposure":
            metric = "exposure"
            out[f"{prefix}exposure_score"] = r.get("exposure_score", 0.5)
            self._put_if_exists(
                out,
                prefix,
                r,
                (
                    "mean_brightness",
                    "mean_brightness_8bit",
                    "exposure_grade",
                    "exposure_eval_status",
                    "exposure_fallback_reason",
                ),
            )
            self._finalize_metric(out, prefix, metric, ensure_grade=True)
            return out

        # -------------------------
        # local_sharpness / local_contrast
        # -------------------------
        if name in ("local_sharpness", "local_contrast"):
            metric = name
            out[f"{prefix}{name}_score"] = r.get(f"{name}_score", 0)
            self._put_if_exists(
                out,
                prefix,
                r,
                (
                    f"{name}_raw",
                    f"{name}_std",
                    f"{name}_eval_status",
                    f"{name}_fallback_reason",
                ),
            )
            # local系は grade を持たない設計でOK（補完しない）
            self._finalize_metric(out, prefix, metric, ensure_grade=False)
            return out

        # -------------------------
        # default: 1 score only
        # -------------------------
        out[f"{prefix}{name}_score"] = r.get(f"{name}_score", 0)
        return out
