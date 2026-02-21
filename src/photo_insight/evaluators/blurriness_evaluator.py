from __future__ import annotations

import cv2
import numpy as np

from photo_insight.utils.image_utils import ImageUtils


class BlurrinessEvaluator:
    """
    ぼやけ具合（= シャープさ）を評価するクラス（複数特徴量を統合）

    Contract (fixed):
      - blurriness_raw: 連続値。大きいほどシャープ（ピントが合っている） = higher is better
      - blurriness_raw_direction: "higher_is_better" 固定（欠損しない）
      - blurriness_raw_transform: "identity" 固定（欠損しない）
      - blurriness_higher_is_better: True 固定（欠損しない）

    Backward compatible outputs:
      - blurriness_score: 0,0.25,0.5,0.75,1.0 の5段階
      - blurriness_grade: "very_blurry" 〜 "excellent"
      - blurriness_eval_status: "ok" / "fallback" / "invalid"
      - blurriness_fallback_reason: fallback/invalid の理由（ok は ""）
      - variance_of_* は解析用の詳細として残す

    Notes:
      - 入力画像の dtype/スケール差を吸収するため、必ずグレースケールfloat32[0..1]に正規化してから計算する。
      - Sobel / Laplacian / Gaussian-diff の複数特徴を統合するのはOK（頑健性が上がる）。
    """

    # ---- contract constants ----
    RAW_DIRECTION = "higher_is_better"
    RAW_TRANSFORM = "identity"
    HIGHER_IS_BETTER = True

    DEFAULT_DISCRETIZE_THRESHOLDS_RAW = {
        # “保険”のデフォルト（実データで調整される前提）
        # ※0..1 正規化後の raw スケール前提。必ず score_dist_tune で再チューニングすること。
        "bad": 1e-5,
        "poor": 5e-5,
        "fair": 2e-4,
        "good": 8e-4,
    }

    def __init__(
        self,
        grad_weight: float = 0.4,
        lap_weight: float = 0.4,
        diff_weight: float = 0.2,
        min_size: int = 64,
        logger=None,
        config=None,
        # feature params (optional)
        sobel_ksize: int = 3,
        laplacian_ksize: int = 3,
        gaussian_ksize: int = 5,
        gaussian_sigma: float = 0.0,
        # robustification
        clip_percentile: float | None = 99.9,
    ) -> None:
        self.grad_weight = float(grad_weight)
        self.lap_weight = float(lap_weight)
        self.diff_weight = float(diff_weight)
        self.min_size = int(min_size)
        self.logger = logger

        self.sobel_ksize = int(sobel_ksize)
        self.laplacian_ksize = int(laplacian_ksize)
        self.gaussian_ksize = int(gaussian_ksize)
        self.gaussian_sigma = float(gaussian_sigma)
        self.clip_percentile = (
            float(clip_percentile) if clip_percentile is not None else None
        )

        # --- normalize weights to avoid future accidents ---
        ws = np.array(
            [self.grad_weight, self.lap_weight, self.diff_weight], dtype=np.float32
        )
        s = float(ws.sum())
        if s > 0:
            ws = ws / s
        self.grad_weight, self.lap_weight, self.diff_weight = map(float, ws)

        # --- config thresholds ---
        cfg = config or {}
        blur_cfg = cfg.get("blurriness", {}) if isinstance(cfg, dict) else {}

        thresholds = blur_cfg.get("discretize_thresholds_raw", {})
        if not isinstance(thresholds, dict):
            thresholds = {}

        def _get(name: str) -> float:
            v = thresholds.get(name, self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW[name])
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW[name])

        # 4境界: bad/poor/fair/good
        self.t_bad = _get("bad")
        self.t_poor = _get("poor")
        self.t_fair = _get("fair")
        self.t_good = _get("good")

        # 単調性崩れの事故防止（並べ直し）
        ts = sorted([self.t_bad, self.t_poor, self.t_fair, self.t_good])
        self.t_bad, self.t_poor, self.t_fair, self.t_good = ts

        # optional param overrides from config
        # (safe: ignore if missing / wrong type)
        try:
            feature_cfg = (
                blur_cfg.get("feature_params", {}) if isinstance(blur_cfg, dict) else {}
            )
            if isinstance(feature_cfg, dict):
                self.sobel_ksize = int(feature_cfg.get("sobel_ksize", self.sobel_ksize))
                self.laplacian_ksize = int(
                    feature_cfg.get("laplacian_ksize", self.laplacian_ksize)
                )
                self.gaussian_ksize = int(
                    feature_cfg.get("gaussian_ksize", self.gaussian_ksize)
                )
                self.gaussian_sigma = float(
                    feature_cfg.get("gaussian_sigma", self.gaussian_sigma)
                )
                cp = feature_cfg.get("clip_percentile", self.clip_percentile)
                self.clip_percentile = float(cp) if cp is not None else None
        except Exception:
            pass

        # ksize sanity
        if self.sobel_ksize not in (1, 3, 5, 7):
            self.sobel_ksize = 3
        if self.laplacian_ksize not in (1, 3, 5, 7):
            self.laplacian_ksize = 3
        if self.gaussian_ksize % 2 == 0 or self.gaussian_ksize < 3:
            self.gaussian_ksize = 5

        if self.logger is not None:
            try:
                self.logger.debug(
                    "[BlurrinessEvaluator] discretize_thresholds_raw="
                    f"bad:{self.t_bad}, poor:{self.t_poor}, fair:{self.t_fair}, "
                    f"good:{self.t_good}"
                )
                self.logger.debug(
                    "[BlurrinessEvaluator] raw_contract="
                    f"direction:{self.RAW_DIRECTION}, transform:{self.RAW_TRANSFORM}, "
                    f"higher_is_better:{self.HIGHER_IS_BETTER}"
                )
                self.logger.debug(
                    "[BlurrinessEvaluator] weights="
                    f"grad:{self.grad_weight:.3f}, lap:{self.lap_weight:.3f}, "
                    f"diff:{self.diff_weight:.3f}"
                    f" | ksizes sobel:{self.sobel_ksize}, lap:{self.laplacian_ksize}, "
                    f"gauss:{self.gaussian_ksize}"
                    f" | clip_percentile:{self.clip_percentile}"
                )
            except Exception:
                pass

    # -------------------------
    # helpers
    # -------------------------
    def _to_gray01(self, image: np.ndarray) -> np.ndarray:
        """Return grayscale float32 in [0, 1]."""
        gray = ImageUtils.to_grayscale(image) if len(image.shape) == 3 else image

        if not isinstance(gray, np.ndarray):
            raise TypeError("gray_image_not_ndarray")

        if gray.dtype == np.uint8:
            gray01 = gray.astype(np.float32) / 255.0
        elif gray.dtype == np.uint16:
            gray01 = gray.astype(np.float32) / 65535.0
        else:
            gray01 = gray.astype(np.float32)
            # heuristic: if looks like 0..255 floats, normalize
            mx = float(np.nanmax(gray01)) if gray01.size else 0.0
            if mx > 1.5:
                gray01 = gray01 / 255.0

        gray01 = np.clip(gray01, 0.0, 1.0)
        return gray01

    def _robust_clip(self, x: np.ndarray) -> np.ndarray:
        """Clip extreme values to reduce outlier domination (optional)."""
        if self.clip_percentile is None:
            return x
        try:
            p = float(self.clip_percentile)
            if not (0.0 < p < 100.0):
                return x
            hi = float(np.nanpercentile(x, p))
            if not np.isfinite(hi) or hi <= 0:
                return x
            return np.clip(x, 0.0, hi)
        except Exception:
            return x

    def _to_score_and_grade(self, raw: float) -> tuple[float, str]:
        # raw: 大きいほど良い（シャープ）
        if (not np.isfinite(raw)) or raw <= 0.0:
            return 0.0, "very_blurry"

        if raw < self.t_bad:
            return 0.0, "very_blurry"
        elif raw < self.t_poor:
            return 0.25, "blurry"
        elif raw < self.t_fair:
            return 0.5, "fair"
        elif raw < self.t_good:
            return 0.75, "good"
        else:
            return 1.0, "excellent"

    def _result_base(self) -> dict:
        # 欠損しない“契約”の形に固定
        return {
            "blurriness_score": 0.0,
            "blurriness_raw": 0.0,
            "blurriness_grade": "very_blurry",
            "blurriness_eval_status": "invalid",
            "blurriness_fallback_reason": "",
            # ---- contract keys (ALWAYS present) ----
            "blurriness_raw_direction": self.RAW_DIRECTION,
            "blurriness_raw_transform": self.RAW_TRANSFORM,
            "blurriness_higher_is_better": self.HIGHER_IS_BETTER,
            # 解析用詳細（残す）
            "variance_of_gradient": 0.0,
            "variance_of_laplacian": 0.0,
            "variance_of_difference": 0.0,
        }

    # -------------------------
    # main
    # -------------------------
    def evaluate(self, image: np.ndarray) -> dict:
        result = self._result_base()

        # --- 入力チェック（invalid）---
        if not isinstance(image, np.ndarray):
            result["blurriness_eval_status"] = "invalid"
            result["blurriness_fallback_reason"] = "invalid_input_not_ndarray"
            return result
        if image.size == 0:
            result["blurriness_eval_status"] = "invalid"
            result["blurriness_fallback_reason"] = "invalid_input_empty"
            return result

        try:
            gray01 = self._to_gray01(image)
            h, w = gray01.shape[:2]

            eval_status = "ok"
            fallback_reason = ""
            if h < self.min_size or w < self.min_size:
                eval_status = "fallback"
                fallback_reason = f"too_small_image_{w}x{h}"

            # --- Tenengrad (Sobel magnitude) ---
            sobel_x = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
            sobel_y = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=self.sobel_ksize)
            grad_mag = cv2.magnitude(sobel_x, sobel_y)
            grad_mag = self._robust_clip(grad_mag)
            variance_of_gradient = float(np.var(grad_mag))

            # --- Laplacian variance ---
            lap = cv2.Laplacian(gray01, cv2.CV_32F, ksize=self.laplacian_ksize)
            lap = self._robust_clip(lap)
            variance_of_laplacian = float(np.var(lap))

            # --- Gaussian difference variance ---
            blurred = cv2.GaussianBlur(
                gray01, (self.gaussian_ksize, self.gaussian_ksize), self.gaussian_sigma
            )
            diff = cv2.absdiff(gray01, blurred)
            diff = self._robust_clip(diff)
            variance_of_difference = float(np.var(diff))

            # --- integrated raw (higher is better) ---
            blurriness_raw = (
                self.grad_weight * variance_of_gradient
                + self.lap_weight * variance_of_laplacian
                + self.diff_weight * variance_of_difference
            )

            if not np.isfinite(blurriness_raw):
                result["blurriness_eval_status"] = "fallback"
                result["blurriness_fallback_reason"] = "non_finite_raw"
                return result

            blurriness_score, blurriness_grade = self._to_score_and_grade(
                float(blurriness_raw)
            )

            result.update(
                {
                    "blurriness_score": float(blurriness_score),
                    "blurriness_raw": float(blurriness_raw),
                    "blurriness_grade": blurriness_grade,
                    "blurriness_eval_status": eval_status,
                    "blurriness_fallback_reason": fallback_reason,
                    "variance_of_gradient": float(variance_of_gradient),
                    "variance_of_laplacian": float(variance_of_laplacian),
                    "variance_of_difference": float(variance_of_difference),
                    # contract keys 念押し
                    "blurriness_raw_direction": self.RAW_DIRECTION,
                    "blurriness_raw_transform": self.RAW_TRANSFORM,
                    "blurriness_higher_is_better": self.HIGHER_IS_BETTER,
                }
            )
            return result

        except Exception as e:
            result["blurriness_eval_status"] = "fallback"
            result["blurriness_fallback_reason"] = f"exception:{type(e).__name__}"
            return result
