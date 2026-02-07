import numpy as np
from utils.image_utils import ImageUtils

from evaluators.common.raw_contract_utils import ensure_gray255, load_thresholds_sorted


class ContrastEvaluator:
    """
    画像のコントラストを評価するクラス。

    Contract:
      - contrast_raw: grayscale stddev (0..255想定), higher is better
      - raw_direction: "higher_is_better" (固定)
      - raw_transform: "identity" (固定; 将来 normalize 等に拡張可)
      - thresholds: poor/fair/good/excellent を raw 閾値として扱う
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

    def __init__(self, logger=None, config=None) -> None:
        self.logger = logger

        # ★ 共通化：閾値のロードと単調性保証
        ts = load_thresholds_sorted(
            config,
            metric_key="contrast",
            defaults=self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW,
            names_in_order=("poor", "fair", "good", "excellent"),
        )
        self.t_poor = ts["poor"]
        self.t_fair = ts["fair"]
        self.t_good = ts["good"]
        self.t_excellent = ts["excellent"]

        if self.logger is not None:
            try:
                self.logger.debug(
                    "[ContrastEvaluator] discretize_thresholds_raw="
                    f"poor:{self.t_poor}, fair:{self.t_fair}, good:{self.t_good}, excellent:{self.t_excellent}"
                )
            except Exception:
                pass

    def _base_payload(self) -> dict:
        """
        常に返す契約情報（#701系と揃える用）
        """
        return {
            "contrast_raw_direction": self.RAW_DIRECTION,
            "contrast_raw_transform": self.RAW_TRANSFORM,
            "contrast_raw_transform_spec": {
                "name": self.RAW_TRANSFORM,
                "params": {},
            },
            "contrast_thresholds_raw": {
                "poor": float(self.t_poor),
                "fair": float(self.t_fair),
                "good": float(self.t_good),
                "excellent": float(self.t_excellent),
            },
        }

    def evaluate(self, image: np.ndarray) -> dict:
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
        contrast_raw = float(np.std(gray255))

        # ほぼ真っ平・壊れた画像など
        if not np.isfinite(contrast_raw) or contrast_raw <= 0.0:
            out = {
                "contrast_raw": float(contrast_raw),
                "contrast_score": 0.0,
                "contrast_grade": "bad",
                "contrast_eval_status": "invalid_input",
                "contrast_fallback_reason": "non_finite_or_non_positive_raw",
                "success": False,
            }
            out.update(self._base_payload())
            return out

        # 5段階離散化（higher is better）
        if contrast_raw >= self.t_excellent:
            contrast_score = 1.0
            contrast_grade = "excellent"
        elif contrast_raw >= self.t_good:
            contrast_score = 0.75
            contrast_grade = "good"
        elif contrast_raw >= self.t_fair:
            contrast_score = 0.5
            contrast_grade = "fair"
        elif contrast_raw >= self.t_poor:
            contrast_score = 0.25
            contrast_grade = "poor"
        else:
            contrast_score = 0.0
            contrast_grade = "bad"

        out = {
            "contrast_raw": contrast_raw,
            "contrast_score": float(contrast_score),
            "contrast_grade": contrast_grade,
            "contrast_eval_status": "ok",
            "contrast_fallback_reason": "",
            "success": True,
        }
        out.update(self._base_payload())
        return out
