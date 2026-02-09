from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class MetricResultMapper:
    """
    evaluator の生結果(result dict)を、CSV用の標準キーへマッピングする共通化レイヤ。
    prefix="" -> global
    prefix="face_" -> face region
    """

    def map(self, name: str, result: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        r = result or {}
        out: Dict[str, Any] = {}

        # -------------------------
        # Noise
        # -------------------------
        if name == "noise":
            # score は常に出す（欠損時ニュートラル）
            out[f"{prefix}noise_score"] = r.get("noise_score", 0.5)

            # 必要な列だけ拾う（globalでもfaceでも同じ挙動に統一）
            for k in (
                "noise_raw",
                "noise_grade",
                "noise_sigma_midtone",
                "noise_sigma_used",
                "noise_mask_ratio",
                "noise_eval_status",
                "noise_fallback_reason",
                "noise_score_brightness_adjusted",
            ):
                if k in r:
                    out[f"{prefix}{k}"] = r.get(k)

            # 保険：NoiseEvaluator が noise_raw を返してない場合だけ補完
            if f"{prefix}noise_raw" not in out:
                sigma_used = r.get("noise_sigma_used")
                try:
                    out[f"{prefix}noise_raw"] = -float(sigma_used) if sigma_used is not None else None
                except (TypeError, ValueError):
                    out[f"{prefix}noise_raw"] = None

            return out

        # -------------------------
        # Sharpness
        # -------------------------
        if name == "sharpness":
            out[f"{prefix}sharpness_score"] = r.get("sharpness_score", 0.5)
            if "sharpness_raw" in r:
                out[f"{prefix}sharpness_raw"] = r.get("sharpness_raw")
            if "sharpness_eval_status" in r:
                out[f"{prefix}sharpness_eval_status"] = r.get("sharpness_eval_status")
            if "sharpness_fallback_reason" in r:
                out[f"{prefix}sharpness_fallback_reason"] = r.get("sharpness_fallback_reason")
            return out

        # -------------------------
        # Contrast
        # -------------------------
        if name == "contrast":
            out[f"{prefix}contrast_score"] = r.get("contrast_score", 0.5)
            if "contrast_raw" in r:
                out[f"{prefix}contrast_raw"] = r.get("contrast_raw")
            if "contrast_grade" in r:
                out[f"{prefix}contrast_grade"] = r.get("contrast_grade")
            if "contrast_eval_status" in r:
                out[f"{prefix}contrast_eval_status"] = r.get("contrast_eval_status")
            if "contrast_fallback_reason" in r:
                out[f"{prefix}contrast_fallback_reason"] = r.get("contrast_fallback_reason")

            # （任意）追跡性を上げたいなら最低限これだけ拾う
            if "contrast_raw_direction" in r:
                out[f"{prefix}contrast_raw_direction"] = r.get("contrast_raw_direction")
            if "contrast_raw_transform" in r:
                out[f"{prefix}contrast_raw_transform"] = r.get("contrast_raw_transform")
            if "contrast_thresholds_metric_key" in r:
                out[f"{prefix}contrast_thresholds_metric_key"] = r.get("contrast_thresholds_metric_key")

            return out

        # -------------------------
        # Blurriness
        # -------------------------
        if name == "blurriness":
            out[f"{prefix}blurriness_score"] = r.get("blurriness_score", 0.5)
            if "blurriness_raw" in r:
                out[f"{prefix}blurriness_raw"] = r.get("blurriness_raw")
            if "blurriness_grade" in r:
                out[f"{prefix}blurriness_grade"] = r.get("blurriness_grade")
            if "blurriness_eval_status" in r:
                out[f"{prefix}blurriness_eval_status"] = r.get("blurriness_eval_status")
            if "blurriness_fallback_reason" in r:
                out[f"{prefix}blurriness_fallback_reason"] = r.get("blurriness_fallback_reason")
            if "blurriness_score_brightness_adjusted" in r:
                out[f"{prefix}blurriness_score_brightness_adjusted"] = r.get(
                    "blurriness_score_brightness_adjusted"
                )
            return out

        # -------------------------
        # Exposure
        # -------------------------
        if name == "exposure":
            out[f"{prefix}exposure_score"] = r.get("exposure_score", 0.5)
            if "mean_brightness" in r:
                out[f"{prefix}mean_brightness"] = r.get("mean_brightness")
            if "mean_brightness_8bit" in r:
                out[f"{prefix}mean_brightness_8bit"] = r.get("mean_brightness_8bit")
            if "exposure_grade" in r:
                out[f"{prefix}exposure_grade"] = r.get("exposure_grade")
            if "exposure_eval_status" in r:
                out[f"{prefix}exposure_eval_status"] = r.get("exposure_eval_status")
            if "exposure_fallback_reason" in r:
                out[f"{prefix}exposure_fallback_reason"] = r.get("exposure_fallback_reason")
            return out

        # -------------------------
        # local_sharpness / local_contrast
        # -------------------------
        if name in ("local_sharpness", "local_contrast"):
            out[f"{prefix}{name}_score"] = r.get(f"{name}_score", 0)
            if f"{name}_raw" in r:
                out[f"{prefix}{name}_raw"] = r.get(f"{name}_raw")
            if f"{name}_std" in r:
                out[f"{prefix}{name}_std"] = r.get(f"{name}_std", 0)
            if f"{name}_eval_status" in r:
                out[f"{prefix}{name}_eval_status"] = r.get(f"{name}_eval_status")
            if f"{name}_fallback_reason" in r:
                out[f"{prefix}{name}_fallback_reason"] = r.get(f"{name}_fallback_reason")
            return out

        # -------------------------
        # default: 1 score only
        # -------------------------
        out[f"{prefix}{name}_score"] = r.get(f"{name}_score", 0)
        return out
