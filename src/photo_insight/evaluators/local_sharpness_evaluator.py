from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, Optional

from photo_insight.utils.image_utils import ImageUtils
from photo_insight.utils.app_logger import Logger


class LocalSharpnessEvaluator:
    """
    画像の局所シャープネスを評価するクラス（SharpnessEvaluatorと同一契約）。

    出力:
      - local_sharpness_raw         : 代表raw値（mean+p80ブレンド）
      - local_sharpness_score       : 0/0.25/0.5/0.75/1.0 の離散スコア
      - local_sharpness_std         : パッチ間ばらつき
      - local_sharpness_eval_status : ok / invalid_input / error
      - local_sharpness_fallback_reason（任意・推奨）
    """

    RAW_KEY = "local_sharpness_raw"
    SCORE_KEY = "local_sharpness_score"
    STD_KEY = "local_sharpness_std"
    STATUS_KEY = "local_sharpness_eval_status"
    FALLBACK_KEY = "local_sharpness_fallback_reason"

    def __init__(
        self,
        logger: Optional[Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.logger = logger or Logger(logger_name="LocalSharpnessEvaluator")
        self.config = config or {}

        conf = self.config.get("local_sharpness", {})

        self.patch_size: int = int(conf.get("patch_size", 32))
        self.stride: int = int(conf.get("stride", self.patch_size))
        self.min_image_side: int = int(conf.get("min_image_side", 128))
        self.max_patches: int = int(conf.get("max_patches", 0))  # 0 = no limit
        self.laplacian_ksize: int = int(conf.get("laplacian_ksize", 3))

        # 低テクスチャ除外（任意）
        tf = conf.get("texture_filter", {}) or {}
        self.texture_enabled: bool = bool(tf.get("enabled", True))
        self.texture_method: str = str(tf.get("method", "sobel"))
        self.edge_threshold: float = float(tf.get("edge_threshold", 30.0))
        self.min_edge_ratio: float = float(tf.get("min_edge_ratio", 0.02))

        # 離散化閾値（SharpnessEvaluatorと同形式）
        disc = conf.get("discretize_thresholds_raw", {}) or {}
        self.threshold_poor: float = float(disc.get("poor", 8.0))
        self.threshold_fair: float = float(disc.get("fair", 12.0))
        self.threshold_good: float = float(disc.get("good", 18.0))
        self.threshold_excellent: float = float(disc.get("excellent", 26.0))

        # 代表値（平均と分位のブレンド）
        self.p_quantile: float = float(conf.get("representative_quantile", 0.80))
        self.quantile_weight: float = float(
            conf.get("representative_quantile_weight", 0.7)
        )
        self.mean_weight: float = float(conf.get("representative_mean_weight", 0.3))

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            self.logger.warning(
                "LocalSharpnessEvaluator: invalid image. fallback to neutral."
            )
            return {
                self.RAW_KEY: None,
                self.SCORE_KEY: 0.5,
                self.STD_KEY: 0.0,
                self.STATUS_KEY: "invalid_input",
                self.FALLBACK_KEY: "invalid_input",
            }

        try:
            # grayscale + uint8 正規化
            gray = ImageUtils.to_grayscale(image) if image.ndim == 3 else image
            h, w = gray.shape[:2]

            if min(h, w) < self.min_image_side:
                return {
                    self.RAW_KEY: None,
                    self.SCORE_KEY: 0.5,
                    self.STD_KEY: 0.0,
                    self.STATUS_KEY: "invalid_input",
                    self.FALLBACK_KEY: "too_small_image",
                }

            gray_u8 = self._to_u8(gray)

            # Laplacianは全体で1回（高速）
            lap = cv2.Laplacian(gray_u8, cv2.CV_64F, ksize=self.laplacian_ksize)

            edge_mask = self._edge_mask(gray_u8) if self.texture_enabled else None

            values = []
            edge_ratios = []

            patch_count = 0
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    if edge_mask is not None:
                        em = edge_mask[y : y + self.patch_size, x : x + self.patch_size]
                        ratio = float(np.mean(em > 0))
                        edge_ratios.append(ratio)
                        if ratio < self.min_edge_ratio:
                            continue

                    lap_patch = lap[y : y + self.patch_size, x : x + self.patch_size]
                    values.append(float(lap_patch.var()))
                    patch_count += 1

                    if self.max_patches > 0 and patch_count >= self.max_patches:
                        break
                if self.max_patches > 0 and patch_count >= self.max_patches:
                    break

            if not values:
                # 低テクスチャ除外などで実効パッチがゼロ
                return {
                    self.RAW_KEY: None,
                    self.SCORE_KEY: 0.5,
                    self.STD_KEY: 0.0,
                    self.STATUS_KEY: "ok",
                    self.FALLBACK_KEY: "no_effective_patches",
                    "local_sharpness_patch_count": 0,
                    "local_sharpness_edge_ratio_mean": (
                        float(np.mean(edge_ratios)) if edge_ratios else 0.0
                    ),
                }

            arr = np.array(values, dtype=np.float64)
            mean_v = float(arr.mean())
            std_v = float(arr.std())
            q_v = float(np.quantile(arr, self.p_quantile))

            # 代表raw（平均と上位分位のブレンド）
            rep_raw = self.quantile_weight * q_v + self.mean_weight * mean_v

            # 離散スコア（ここが今回の変更点：SCORE_KEYは離散）
            score = self._to_discrete_score(rep_raw)

            result: Dict[str, Any] = {
                self.RAW_KEY: rep_raw,
                self.SCORE_KEY: score,
                self.STD_KEY: std_v,
                self.STATUS_KEY: "ok",
                # デバッグ/分析用（ヘッダ無しならCSV出力側で無視してOK）
                "local_sharpness_mean": mean_v,
                "local_sharpness_p80": q_v,
                "local_sharpness_patch_count": int(arr.size),
            }
            if edge_ratios:
                result["local_sharpness_edge_ratio_mean"] = float(np.mean(edge_ratios))

            self.logger.debug(
                f"local_sharpness_raw={rep_raw:.3f}, local_sharpness_score={score:.2f}, "
                f"mean={mean_v:.3f}, p{int(self.p_quantile*100)}={q_v:.3f}, std={std_v:.3f}, patches={arr.size}"
            )
            return result

        except Exception as e:
            self.logger.warning(
                f"LocalSharpnessEvaluator: exception during evaluate: {type(e).__name__}: {e}"
            )
            return {
                self.RAW_KEY: None,
                self.SCORE_KEY: 0.5,
                self.STD_KEY: 0.0,
                self.STATUS_KEY: "error",
                self.FALLBACK_KEY: "exception",
            }

    def _to_u8(self, gray: np.ndarray) -> np.ndarray:
        if gray.dtype == np.uint8:
            return gray
        g = gray.astype(np.float32)
        g_min = float(g.min())
        g_max = float(g.max())
        if g_max > g_min:
            norm = (g - g_min) / (g_max - g_min)
            return (norm * 255.0).clip(0, 255).astype(np.uint8)
        return np.zeros_like(gray, dtype=np.uint8)

    def _edge_mask(self, gray_u8: np.ndarray) -> np.ndarray:
        # sobel magnitude threshold
        if self.texture_method.lower() == "sobel":
            gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            return (mag >= self.edge_threshold).astype(np.uint8)
        # fallback: canny
        edges = cv2.Canny(gray_u8, threshold1=50, threshold2=150)
        return (edges > 0).astype(np.uint8)

    def _to_discrete_score(self, raw: float) -> float:
        v = float(raw)
        if v >= self.threshold_excellent:
            return 1.0
        if v >= self.threshold_good:
            return 0.75
        if v >= self.threshold_fair:
            return 0.5
        if v >= self.threshold_poor:
            return 0.25
        return 0.0
