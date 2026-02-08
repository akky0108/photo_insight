import numpy as np
from utils.image_utils import ImageUtils

from evaluators.common.raw_contract_utils import ensure_gray255, load_thresholds_sorted


class ContrastEvaluator:
    """
    画像のコントラストを評価するクラス。

    Contract (per metric_key):
      - {metric}_raw        : grayscale stddev (0..255想定), higher is better
      - {metric}_raw_direction: "higher_is_better" (固定)
      - {metric}_raw_transform: "identity" (固定; 将来 normalize 等に拡張可)
      - {metric}_thresholds_raw: poor/fair/good/excellent を raw 閾値として扱う
      - {metric}_score      : 0, 0.25, 0.5, 0.75, 1.0
      - {metric}_grade      : bad/poor/fair/good/excellent
      - {metric}_eval_status: ok / invalid_input
      - {metric}_fallback_reason: "" or reason string

    metric_key:
      - "contrast" (default)
      - "face_contrast" (for face region thresholds separation)
    """

    DEFAULT_DISCRETIZE_THRESHOLDS_RAW = {
        # “保険”のデフォルト（実データで調整される前提）
        "poor": 15.0,
        "fair": 30.0,
        "good": 50.0,
        "excellent": 70.0,
    }

    RAW_DIRECTION = "higher_is_better"
    RAW_TRANSFORM = "identity"

    def __init__(self, logger=None, config=None, metric_key: str = "contrast") -> None:
        self.logger = logger
        self.metric_key = str(metric_key or "contrast")

        # 出力キー prefix（contrast / face_contrast）
        self.out_key = self.metric_key

        # ★ 共通化：閾値のロードと単調性保証
        ts = load_thresholds_sorted(
            config=config,
            metric_key=self.metric_key,
            defaults=self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW,
            names_in_order=("poor", "fair", "good", "excellent"),
        )
        self.t_poor = float(ts["poor"])
        self.t_fair = float(ts["fair"])
        self.t_good = float(ts["good"])
        self.t_excellent = float(ts["excellent"])

        if self.logger is not None:
            try:
                self.logger.debug(
                    f"[ContrastEvaluator:{self.metric_key}] discretize_thresholds_raw="
                    f"poor:{self.t_poor}, fair:{self.t_fair}, good:{self.t_good}, excellent:{self.t_excellent}"
                )
            except Exception:
                pass

    def _base_payload(self) -> dict:
        """
        常に返す契約情報（#701系と揃える用）
        """
        k = self.out_key
        return {
            f"{k}_raw_direction": self.RAW_DIRECTION,
            f"{k}_raw_transform": self.RAW_TRANSFORM,
            f"{k}_raw_transform_spec": {
                "name": self.RAW_TRANSFORM,
                "params": {},
            },
            f"{k}_thresholds_raw": {
                "poor": float(self.t_poor),
                "fair": float(self.t_fair),
                "good": float(self.t_good),
                "excellent": float(self.t_excellent),
            },
            # 追跡しやすいように（必要なければ削ってOK）
            f"{k}_thresholds_metric_key": self.metric_key,
        }

    def evaluate(self, image: np.ndarray) -> dict:
        k = self.out_key

        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid input: expected a numpy array representing an image.")
        if image.size == 0:
            raise ValueError("Invalid input: empty image.")

        # 画像がカラーの場合はグレースケールに変換（BGR想定）
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # ★ 本筋：0..255 スケールへ吸収してから raw 計算
        #   （float01 / uint16 / float255 どれが来ても tune 閾値と整合する）
        gray255 = ensure_gray255(gray_image)

        # 生の標準偏差（raw）
        raw = float(np.std(gray255))

        # ほぼ真っ平・壊れた画像など
        if not np.isfinite(raw) or raw <= 0.0:
            out = {
                f"{k}_raw": float(raw),
                f"{k}_score": 0.0,
                f"{k}_grade": "bad",
                f"{k}_eval_status": "invalid_input",
                f"{k}_fallback_reason": "non_finite_or_non_positive_raw",
                "success": False,
            }
            out.update(self._base_payload())
            return out

        # 5段階離散化（higher is better）
        if raw >= self.t_excellent:
            score = 1.0
            grade = "excellent"
        elif raw >= self.t_good:
            score = 0.75
            grade = "good"
        elif raw >= self.t_fair:
            score = 0.5
            grade = "fair"
        elif raw >= self.t_poor:
            score = 0.25
            grade = "poor"
        else:
            score = 0.0
            grade = "bad"

        out = {
            f"{k}_raw": float(raw),
            f"{k}_score": float(score),
            f"{k}_grade": grade,
            f"{k}_eval_status": "ok",
            f"{k}_fallback_reason": "",
            "success": True,
        }
        out.update(self._base_payload())
        return out
