# src/evaluators/local_contrast_evaluator.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from photo_insight.utils.image_utils import ImageUtils


@dataclass
class LocalContrastParams:
    # --- local measurement ---
    block_size: int = 32
    min_blocks: int = 9

    # 「ほぼ単色」ブロック除外（比率）
    # 例: (max-min)/max < threshold なら除外
    ignore_low_dynamic_blocks: bool = True
    low_dynamic_threshold: float = 0.02

    # 外れ値耐性（robust aggregate）
    robust_p_low: float = 5.0
    robust_p_high: float = 95.0

    # --- raw -> score normalization (ratio based) ---
    # raw は std/mean の比率（スケール非依存）
    raw_floor: float = 0.010
    raw_ceil: float = 0.060
    gamma: float = 0.9

    # --- numeric stability ---
    eps: float = 1e-6


class LocalContrastEvaluator:
    """
    局所コントラスト評価（スケール非依存版）

    重要:
      - raw は "std/mean" の比率を採用（0..255 / 0..1 で同等になりやすい）
      - score は raw_floor/raw_ceil/gamma により 5段階へ
      - robust_p_low/high で外れ値の影響を抑える
    """

    def __init__(
        self,
        block_size: int = 32,
        min_blocks: int = 9,
        ignore_low_dynamic_blocks: bool = True,
        low_dynamic_threshold: float = 0.02,
        robust_p_low: float = 5.0,
        robust_p_high: float = 95.0,
        raw_floor: float = 0.010,
        raw_ceil: float = 0.060,
        gamma: float = 0.9,
    ) -> None:
        self.params = LocalContrastParams(
            block_size=int(block_size),
            min_blocks=int(min_blocks),
            ignore_low_dynamic_blocks=bool(ignore_low_dynamic_blocks),
            low_dynamic_threshold=float(low_dynamic_threshold),
            robust_p_low=float(robust_p_low),
            robust_p_high=float(robust_p_high),
            raw_floor=float(raw_floor),
            raw_ceil=float(raw_ceil),
            gamma=float(gamma),
        )

    # =====================================================
    # public
    # =====================================================

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        if not isinstance(image, np.ndarray):
            return self._invalid("invalid_input:type_not_ndarray")

        if image.size == 0:
            return self._invalid("invalid_input:empty_image")

        # --- grayscale ---
        try:
            if image.ndim == 3 and image.shape[2] == 3:
                gray = ImageUtils.to_grayscale(image)
            elif image.ndim == 2:
                gray = image
            else:
                return self._invalid("invalid_input:unsupported_shape")
        except Exception as e:
            return self._invalid(f"invalid_input:grayscale_failed:{type(e).__name__}")

        try:
            gray_f = gray.astype(np.float32)
        except Exception as e:
            return self._invalid(f"invalid_input:astype_failed:{type(e).__name__}")

        # safety: reject all-nonfinite early
        if not np.isfinite(gray_f).any():
            return self._invalid("invalid_input:all_nonfinite")

        h, w = gray_f.shape[:2]
        bs = self.params.block_size

        if h < bs or w < bs:
            return self._fallback_to_global_ratio(gray_f, "image_too_small_for_blocks")

        ratios = self._compute_block_contrast_ratios(gray_f)

        if ratios.size < self.params.min_blocks:
            return self._fallback_to_global_ratio(
                gray_f, f"insufficient_blocks:{int(ratios.size)}"
            )

        # robust winsorize
        ratios_robust = self._winsorize(
            ratios, self.params.robust_p_low, self.params.robust_p_high
        )

        local_raw = float(np.median(ratios_robust))
        local_std = float(np.std(ratios_robust))

        if (not np.isfinite(local_raw)) or local_raw <= 0.0:
            return self._fallback_to_global_ratio(
                gray_f, "nonfinite_or_nonpositive_local_raw"
            )

        # raw -> [0,1] + gamma
        norm = self._normalize_01(
            local_raw, self.params.raw_floor, self.params.raw_ceil
        )
        norm = float(norm**self.params.gamma)

        score, grade = self._to_discrete_score(norm)

        # optional debug fields (non-breaking; useful for QA/analysis)
        try:
            p_lo = float(np.percentile(ratios, self.params.robust_p_low))
            p_hi = float(np.percentile(ratios, self.params.robust_p_high))
        except Exception:
            p_lo = 0.0
            p_hi = 0.0

        return {
            "local_contrast_raw": local_raw,
            "local_contrast_score": score,
            "local_contrast_std": local_std,
            "local_contrast_eval_status": "ok",
            "local_contrast_fallback_reason": "",
            "success": True,
            "local_contrast_grade": grade,
            "local_contrast_n_blocks": int(ratios.size),
            "local_contrast_raw_p_low": p_lo,
            "local_contrast_raw_p_high": p_hi,
        }

    # =====================================================
    # internals
    # =====================================================

    def _compute_block_contrast_ratios(self, gray_f: np.ndarray) -> np.ndarray:
        """
        ブロックごとに raw_ratio = std / mean を計算して返す（スケール非依存）

        注意:
        - ブロック内に NaN/inf が混入していても評価が破綻しないよう finite のみで統計を取る
        - mean が 0 近傍だと ratio が発散するため除外する（abs(mean) <= eps）
        """
        h, w = gray_f.shape[:2]
        bs = self.params.block_size

        vals: list[float] = []

        for y in range(0, h, bs):
            y2 = y + bs
            if y2 > h:
                continue

            for x in range(0, w, bs):
                x2 = x + bs
                if x2 > w:
                    continue

                block = gray_f[y:y2, x:x2]
                if block.size == 0:
                    continue

                # finite-only stats (avoid NaN poisoning)
                finite = np.isfinite(block)
                if not finite.any():
                    continue

                bf = block[finite]
                if bf.size == 0:
                    continue

                bmax = float(np.max(bf))
                bmin = float(np.min(bf))
                if (not np.isfinite(bmax)) or (not np.isfinite(bmin)):
                    continue

                # if everything is ~0, std/mean becomes unstable;
                # treat as uninformative block.
                if abs(bmax) <= self.params.eps and abs(bmin) <= self.params.eps:
                    continue

                if self.params.ignore_low_dynamic_blocks:
                    # ratio-based dynamic range check
                    denom = max(abs(bmax), self.params.eps)
                    dyn_ratio = (bmax - bmin) / denom
                    if dyn_ratio < self.params.low_dynamic_threshold:
                        continue

                bmean = float(np.mean(bf))
                if (not np.isfinite(bmean)) or abs(bmean) <= self.params.eps:
                    continue

                bstd = float(np.std(bf))
                if (not np.isfinite(bstd)) or bstd <= 0.0:
                    continue

                ratio = bstd / max(abs(bmean), self.params.eps)
                if np.isfinite(ratio) and ratio > 0.0:
                    vals.append(float(ratio))

        if not vals:
            return np.asarray([], dtype=np.float32)

        return np.asarray(vals, dtype=np.float32)

    @staticmethod
    def _winsorize(x: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
        """
        外れ値をパーセンタイルでクリップして安定化する。
        """
        if x.size == 0:
            return x

        lo = float(np.percentile(x, p_low))
        hi = float(np.percentile(x, p_high))

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return x

        return np.clip(x, lo, hi)

    @staticmethod
    def _normalize_01(raw: float, raw_floor: float, raw_ceil: float) -> float:
        if raw_ceil <= raw_floor:
            return 1.0

        norm = (raw - raw_floor) / (raw_ceil - raw_floor)
        if norm < 0.0:
            norm = 0.0
        elif norm > 1.0:
            norm = 1.0
        return float(norm)

    @staticmethod
    def _to_discrete_score(norm_01: float) -> Tuple[float, str]:
        n = float(norm_01)
        if n >= 0.85:
            return 1.0, "excellent"
        if n >= 0.70:
            return 0.75, "good"
        if n >= 0.50:
            return 0.5, "fair"
        if n >= 0.30:
            return 0.25, "poor"
        return 0.0, "bad"

    # -----------------------------------------------------

    def _invalid(self, reason: str) -> Dict[str, Any]:
        return {
            "local_contrast_raw": 0.0,
            "local_contrast_score": 0.0,
            "local_contrast_std": 0.0,
            "local_contrast_eval_status": "invalid_input",
            "local_contrast_fallback_reason": reason,
            "success": False,
            "local_contrast_grade": "bad",
            "local_contrast_n_blocks": 0,
        }

    def _fallback_to_global_ratio(
        self, gray_f: np.ndarray, reason: str
    ) -> Dict[str, Any]:
        """
        ブロック計算できない場合のフォールバック：
        グローバル std/mean を raw として扱う（スケール非依存）

        注意:
        - NaN/inf 混入に耐性を持たせるため finite のみで統計を取る
        - mean が 0 近傍だと ratio が発散するため invalid に落とす
        """
        try:
            finite = np.isfinite(gray_f)
            if not finite.any():
                return self._invalid("fallback_failed:all_nonfinite")

            gf = gray_f[finite]
            mean = float(np.mean(gf))
            std = float(np.std(gf))
        except Exception as e:
            return self._invalid(f"fallback_failed:{type(e).__name__}")

        if (not np.isfinite(mean)) or abs(mean) <= self.params.eps:
            return self._invalid("fallback_failed:nonfinite_or_nonpositive_mean")

        if (not np.isfinite(std)) or std <= 0.0:
            return self._invalid("fallback_failed:nonfinite_or_nonpositive_std")

        raw = std / max(abs(mean), self.params.eps)

        norm = self._normalize_01(raw, self.params.raw_floor, self.params.raw_ceil)
        norm = float(norm**self.params.gamma)
        score, grade = self._to_discrete_score(norm)

        return {
            "local_contrast_raw": float(raw),
            "local_contrast_score": score,
            "local_contrast_std": 0.0,
            "local_contrast_eval_status": "fallback_used",
            "local_contrast_fallback_reason": reason,
            "success": True,
            "local_contrast_grade": grade,
            "local_contrast_n_blocks": 0,
        }
