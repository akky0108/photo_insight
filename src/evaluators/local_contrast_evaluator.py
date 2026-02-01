# src/evaluators/local_contrast_evaluator.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from utils.image_utils import ImageUtils


@dataclass
class LocalContrastParams:
    # --- local measurement ---
    block_size: int = 32                 # ブロック（タイル）サイズ
    min_blocks: int = 9                  # 最低ブロック数（例: 3x3）
    ignore_low_dynamic_blocks: bool = True

    # ★ スケール非依存（推奨）
    # (block max-min) / robust_range がこれ未満のブロックは“ほぼ単色”として除外
    low_dynamic_threshold: float = 0.02

    # --- robust range (for scale-invariant normalization) ---
    robust_p_low: float = 5.0
    robust_p_high: float = 95.0

    # --- raw -> score normalization ---
    # ★ 重要: ここでいう raw_floor/raw_ceil は「local_raw / robust_range（比率）」のレンジ
    # つまり入力が 0..1 / 0..255 どちらでも score が安定するようにする。
    raw_floor: float = 0.010
    raw_ceil: float = 0.060
    gamma: float = 0.9

    # --- numeric stability ---
    eps: float = 1e-6


class LocalContrastEvaluator:
    """
    画像の局所コントラストを評価するクラス（LocalSharpness と同一ノリの契約）。

    出力:
      - local_contrast_raw             : ブロック内 std の代表値（中央値） ※スケール依存（比較はscore推奨）
      - local_contrast_score           : 0/0.25/0.5/0.75/1.0 の 5段階離散（スケール非依存）
      - local_contrast_std             : ブロック std の散らばり（標準偏差）
      - local_contrast_eval_status     : ok / invalid_input / fallback_used
      - local_contrast_fallback_reason : フォールバック理由
      - success                        : bool

    設計意図（事故防止・汎用性）:
      - raw（block std中央値）は従来通り保持（指標固有の量として扱う）
      - scoreは入力スケールに依存しないよう、
        norm_raw = local_raw / robust_range(p95-p05) を作ってから正規化→離散化する
      - これにより、ImageUtils.to_grayscale の返すレンジ（0..1/0..255 等）が変わっても 0固定事故を防ぐ
    """

    def __init__(
        self,
        block_size: int = 32,
        raw_floor: float = 0.010,
        raw_ceil: float = 0.060,
        gamma: float = 0.9,
        min_blocks: int = 9,
        ignore_low_dynamic_blocks: bool = True,
        low_dynamic_threshold: float = 0.02,
        robust_p_low: float = 5.0,
        robust_p_high: float = 95.0,
    ) -> None:
        self.params = LocalContrastParams(
            block_size=int(block_size),
            raw_floor=float(raw_floor),
            raw_ceil=float(raw_ceil),
            gamma=float(gamma),
            min_blocks=int(min_blocks),
            ignore_low_dynamic_blocks=bool(ignore_low_dynamic_blocks),
            low_dynamic_threshold=float(low_dynamic_threshold),
            robust_p_low=float(robust_p_low),
            robust_p_high=float(robust_p_high),
        )

    # ----------------------------
    # public
    # ----------------------------
    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        :param image: 評価対象の画像 (np.ndarray)
        :return: 契約に沿った dict
        """
        if not isinstance(image, np.ndarray):
            return self._invalid("invalid_input:type_not_ndarray")

        # 3ch (BGR/RGB) / 1ch を許容し、utils に寄せる
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

        h, w = gray_f.shape[:2]
        bs = self.params.block_size

        # robust_range は “スコア化” と “低ダイナミック除外” に使用
        robust_r, r_note = self._robust_range(gray_f)
        if robust_r <= 0.0:
            # rangeが取れないと「スケール非依存なscore」が作れないのでフォールバックへ
            return self._fallback_to_global_std(gray_f, f"robust_range_failed:{r_note}")

        if h < bs or w < bs:
            # ブロックを切れない → グローバル std でフォールバック
            return self._fallback_to_global_std(gray_f, "image_too_small_for_blocks")

        block_stds = self._compute_block_stds(gray_f, robust_r)
        if block_stds.size < self.params.min_blocks:
            return self._fallback_to_global_std(gray_f, f"insufficient_blocks:{int(block_stds.size)}")

        local_raw = float(np.median(block_stds))
        local_std = float(np.std(block_stds))

        if (not np.isfinite(local_raw)) or local_raw <= 0.0:
            return self._fallback_to_global_std(gray_f, "nonfinite_or_nonpositive_local_raw")

        # ----------------------------
        # ★ scale-invariant score
        # norm_raw := local_raw / robust_range
        # ----------------------------
        norm_raw = float(local_raw / max(robust_r, self.params.eps))

        # norm_raw -> [0,1] 正規化＋ガンマ補正
        norm = self._normalize_01(norm_raw, self.params.raw_floor, self.params.raw_ceil)
        norm = float(norm ** self.params.gamma)

        score, grade = self._to_discrete_score(norm)

        return {
            "local_contrast_raw": local_raw,   # 指標固有の量（スケール依存）
            "local_contrast_score": score,     # 比率正規化によりスケール非依存
            "local_contrast_std": local_std,
            "local_contrast_eval_status": "ok",
            "local_contrast_fallback_reason": "",
            "success": True,
            "local_contrast_grade": grade,     # optional
        }

    # ----------------------------
    # internals
    # ----------------------------
    def _robust_range(self, gray_f: np.ndarray) -> Tuple[float, str]:
        """
        p95 - p05 のロバストレンジ。入力が 0..1 / 0..255 等でも機能する。
        """
        try:
            p_low = float(np.percentile(gray_f, self.params.robust_p_low))
            p_high = float(np.percentile(gray_f, self.params.robust_p_high))
            r = p_high - p_low
            if (not np.isfinite(r)) or r <= self.params.eps:
                return 0.0, "nonfinite_or_too_small"
            return float(r), "ok"
        except Exception as e:
            return 0.0, f"exception:{type(e).__name__}"

    def _compute_block_stds(self, gray_f: np.ndarray, robust_range: float) -> np.ndarray:
        """
        gray_f を block_size ごとに切り、各ブロックの std を返す。
        端の不足ブロックは現行実装同様スキップ（= 完全ブロックのみ）。

        low_dynamic_threshold は「(max-min)/robust_range の比率」で判定（スケール非依存）。
        """
        h, w = gray_f.shape[:2]
        bs = self.params.block_size
        rr = float(max(robust_range, self.params.eps))

        stds = []
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

                # “ほぼ単色”ブロック（背景・空など）を除外して安定化（任意）
                if self.params.ignore_low_dynamic_blocks:
                    bmax = float(np.max(block))
                    bmin = float(np.min(block))
                    dyn_ratio = (bmax - bmin) / rr
                    if dyn_ratio < self.params.low_dynamic_threshold:
                        continue

                bstd = float(np.std(block))
                if np.isfinite(bstd) and bstd > 0.0:
                    stds.append(bstd)

        if not stds:
            return np.asarray([], dtype=np.float32)
        return np.asarray(stds, dtype=np.float32)

    @staticmethod
    def _normalize_01(raw: float, raw_floor: float, raw_ceil: float) -> float:
        """
        raw を [0,1] に正規化（clamp）。
        """
        rf = float(raw_floor)
        rc = float(raw_ceil)
        r = float(raw)

        if rc <= rf:
            return 1.0

        norm = (r - rf) / (rc - rf)
        if norm < 0.0:
            norm = 0.0
        elif norm > 1.0:
            norm = 1.0
        return float(norm)

    @staticmethod
    def _to_discrete_score(norm_01: float) -> Tuple[float, str]:
        """
        norm_01: 0..1
        """
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

    def _invalid(self, reason: str) -> Dict[str, Any]:
        return {
            "local_contrast_raw": 0.0,
            "local_contrast_score": 0.0,
            "local_contrast_std": 0.0,
            "local_contrast_eval_status": "invalid_input",
            "local_contrast_fallback_reason": reason,
            "success": False,
        }

    def _fallback_to_global_std(self, gray_f: np.ndarray, reason: str) -> Dict[str, Any]:
        """
        ローカル計算ができない場合に、グローバル std を raw として扱う。
        スコア化は「raw / robust_range（比率）」を使って、可能な限りスケール非依存を維持する。
        """
        try:
            raw = float(np.std(gray_f))
        except Exception as e:
            return self._invalid(f"fallback_failed:{type(e).__name__}")

        if (not np.isfinite(raw)) or raw <= 0.0:
            return self._invalid("fallback_failed:nonfinite_or_nonpositive_global_std")

        robust_r, r_note = self._robust_range(gray_f)
        if robust_r <= 0.0:
            # scoreの作りようがないので、bad扱いで理由を残す（silentに0固定にしない）
            return {
                "local_contrast_raw": raw,
                "local_contrast_score": 0.0,
                "local_contrast_std": 0.0,
                "local_contrast_eval_status": "fallback_used",
                "local_contrast_fallback_reason": f"{reason}|robust_range_failed:{r_note}",
                "success": True,
                "local_contrast_grade": "bad",
            }

        norm_raw = float(raw / max(robust_r, self.params.eps))
        norm = self._normalize_01(norm_raw, self.params.raw_floor, self.params.raw_ceil)
        norm = float(norm ** self.params.gamma)
        score, grade = self._to_discrete_score(norm)

        return {
            "local_contrast_raw": raw,
            "local_contrast_score": score,
            "local_contrast_std": 0.0,
            "local_contrast_eval_status": "fallback_used",
            "local_contrast_fallback_reason": reason,
            "success": True,
            "local_contrast_grade": grade,
        }
