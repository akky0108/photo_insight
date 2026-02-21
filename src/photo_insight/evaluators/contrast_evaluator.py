from __future__ import annotations

import numpy as np
from photo_insight.utils.image_utils import ImageUtils

from photo_insight.evaluators.common.raw_contract_utils import (
    ensure_gray255,
    load_thresholds_sorted,
)


class ContrastEvaluator:
    """
    画像のコントラストを評価するクラス。

    Aligned design with other metrics:
      - 出力キーは常に "contrast_*"（face_ の付与は mapper 側の責務）
      - face_contrast の閾値分離は、外側で config 注入することで実現する：
          _subcfg("face_contrast") -> {"contrast": cfg["face_contrast"]}
        よって evaluator は metric_key="contrast" のままで動くのが基本。

    Contract (fixed output keys):
      - contrast_raw              : grayscale stddev (0..255想定), higher is better
      - contrast_score            : 0, 0.25, 0.5, 0.75, 1.0
      - contrast_grade            : bad/poor/fair/good/excellent
      - contrast_eval_status      : ok / fallback / invalid
      - contrast_fallback_reason  : "" or reason string
      - contrast_raw_direction    : "higher_is_better" (always)
      - contrast_raw_transform    : "identity" (always)
      - contrast_raw_transform_spec : {"name": "identity", "params": {}}
      - contrast_thresholds_raw   : {"poor","fair","good","excellent"} (always)
      - contrast_thresholds_metric_key : thresholds lookup key (always)
    """

    DEFAULT_DISCRETIZE_THRESHOLDS_RAW = {
        "poor": 15.0,
        "fair": 30.0,
        "good": 50.0,
        "excellent": 70.0,
    }

    RAW_DIRECTION = "higher_is_better"
    RAW_TRANSFORM = "identity"

    def __init__(self, logger=None, config=None, metric_key: str = "contrast") -> None:
        self.logger = logger

        # 閾値参照キー（通常は "contrast"）
        # 注入方式なら face側も {"contrast": ...} になるので "contrast" のままでOK
        self.metric_key = str(metric_key or "contrast")

        # ★出力キーは base 固定（mapperがface_を付与）
        self.out_key = "contrast"

        ts = load_thresholds_sorted(
            config=config,
            metric_key=self.metric_key,
            defaults=self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW,
            names_in_order=("poor", "fair", "good", "excellent"),
        )

        # 念のため float 化（load_thresholds_sorted が保証していても安全）
        self.t_poor = float(ts["poor"])
        self.t_fair = float(ts["fair"])
        self.t_good = float(ts["good"])
        self.t_excellent = float(ts["excellent"])

        if self.logger is not None:
            try:
                self.logger.debug(
                    f"[ContrastEvaluator thresholds_key={self.metric_key}] "
                    f"poor:{self.t_poor}, "
                    f"fair:{self.t_fair}, "
                    f"good:{self.t_good}, "
                    f"excellent:{self.t_excellent}"
                )
            except Exception:
                pass

    # -------------------------
    # contract payload (always)
    # -------------------------
    def _base_payload(self) -> dict:
        k = self.out_key
        return {
            f"{k}_raw_direction": self.RAW_DIRECTION,
            f"{k}_raw_transform": self.RAW_TRANSFORM,
            f"{k}_raw_transform_spec": {"name": self.RAW_TRANSFORM, "params": {}},
            f"{k}_thresholds_raw": {
                "poor": float(self.t_poor),
                "fair": float(self.t_fair),
                "good": float(self.t_good),
                "excellent": float(self.t_excellent),
            },
            # 追跡用：どのキーで閾値を引いたか（注入方式のデバッグにも効く）
            f"{k}_thresholds_metric_key": self.metric_key,
        }

    def _result_base(self) -> dict:
        """
        例外や invalid を含め、必ずこの形から返す（他指標と統一）。
        """
        k = self.out_key
        out = {
            f"{k}_raw": None,
            f"{k}_score": 0.0,
            f"{k}_grade": "bad",
            f"{k}_eval_status": "invalid",  # default: invalid
            f"{k}_fallback_reason": "",
            "success": False,
        }
        out.update(self._base_payload())
        return out

    # -------------------------
    # main
    # -------------------------
    def evaluate(self, image: np.ndarray) -> dict:
        k = self.out_key
        out = self._result_base()

        # --- input validation (invalid) ---
        if not isinstance(image, np.ndarray):
            out[f"{k}_fallback_reason"] = "invalid_input_not_ndarray"
            return out
        if image.size == 0:
            out[f"{k}_fallback_reason"] = "invalid_input_empty"
            return out

        try:
            # BGR -> gray
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray_image = ImageUtils.to_grayscale(image)
            else:
                gray_image = image

            # 0..255 に吸収して raw 計算
            gray255 = ensure_gray255(gray_image)
            raw = float(np.std(gray255))

            out[f"{k}_raw"] = float(raw)

            # non-finite / non-positive は invalid 扱い（壊れ画像等）
            if not np.isfinite(raw) or raw <= 0.0:
                out[f"{k}_eval_status"] = "invalid"
                out[f"{k}_fallback_reason"] = "non_finite_or_non_positive_raw"
                return out

            # 5段階離散化（higher is better）
            if raw >= self.t_excellent:
                score, grade = 1.0, "excellent"
            elif raw >= self.t_good:
                score, grade = 0.75, "good"
            elif raw >= self.t_fair:
                score, grade = 0.5, "fair"
            elif raw >= self.t_poor:
                score, grade = 0.25, "poor"
            else:
                score, grade = 0.0, "bad"

            out.update(
                {
                    f"{k}_score": float(score),
                    f"{k}_grade": grade,
                    f"{k}_eval_status": "ok",
                    f"{k}_fallback_reason": "",
                    "success": True,
                }
            )
            return out

        except Exception as e:
            # Blurriness と同じ思想：落とさず fallback で返す
            out[f"{k}_eval_status"] = "fallback"
            out[f"{k}_fallback_reason"] = f"exception:{type(e).__name__}"
            try:
                if self.logger is not None:
                    self.logger.error(f"[ContrastEvaluator] evaluate failed: {e}")
            except Exception:
                pass
            return out
