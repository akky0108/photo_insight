import numpy as np
from utils.image_utils import ImageUtils


class ContrastEvaluator:
    """
    画像のコントラストを評価するクラス。
    - contrast_raw        : グレースケール画像の標準偏差 (0〜255 想定)
    - contrast_score      : 0〜1 の 5段階離散スコア (0, 0.25, 0.5, 0.75, 1.0)
    - contrast_grade      : excellent / good / fair / poor / bad
    - contrast_eval_status: ok / invalid_input など
    """

    DEFAULT_DISCRETIZE_THRESHOLDS_RAW = {
        # “保険”のデフォルト（実データで調整される前提）
        "poor": 15.0,
        "fair": 30.0,
        "good": 50.0,
        "excellent": 70.0,
    }

    def __init__(self, logger=None, config=None) -> None:
        """
        config 例:
        {
          "contrast": {
            "discretize_thresholds_raw": {
              "poor": 17.7,
              "fair": 39.2,
              "good": 64.5,
              "excellent": 91.9
            }
          }
        }
        """
        self.logger = logger
        cfg = config or {}
        contrast_cfg = cfg.get("contrast", {}) if isinstance(cfg, dict) else {}

        thresholds = contrast_cfg.get("discretize_thresholds_raw", {})
        if not isinstance(thresholds, dict):
            thresholds = {}

        def _get(name: str) -> float:
            v = thresholds.get(name, self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW[name])
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(self.DEFAULT_DISCRETIZE_THRESHOLDS_RAW[name])

        # t0,t1,t2,t3 を poor/fair/good/excellent として固定
        self.t_poor = _get("poor")
        self.t_fair = _get("fair")
        self.t_good = _get("good")
        self.t_excellent = _get("excellent")

        # 閾値の単調性が崩れていたら安全側に並べ直す（事故防止）
        ts = sorted([self.t_poor, self.t_fair, self.t_good, self.t_excellent])
        self.t_poor, self.t_fair, self.t_good, self.t_excellent = ts

        if self.logger is not None:
            try:
                self.logger.debug(
                    f"[ContrastEvaluator] discretize_thresholds_raw="
                    f"poor:{self.t_poor}, fair:{self.t_fair}, good:{self.t_good}, excellent:{self.t_excellent}"
                )
            except Exception:
                # logger が標準的APIでない可能性もあるので握りつぶす
                pass

    def evaluate(self, image: np.ndarray) -> dict:
        if not isinstance(image, np.ndarray):
            raise ValueError("Invalid input: expected a numpy array representing an image.")

        # 画像がカラーの場合はグレースケールに変換（BGR想定）
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        gray = gray_image.astype(np.float32)

        # 生の標準偏差（raw）
        contrast_raw = float(np.std(gray))

        # ほぼ真っ平・壊れた画像など
        if not np.isfinite(contrast_raw) or contrast_raw <= 0.0:
            return {
                "contrast_raw": float(contrast_raw),
                "contrast_score": 0.0,
                "contrast_grade": "bad",
                "contrast_eval_status": "invalid_input",
                "success": False,
            }

        # ----------------------------
        # raw 閾値による 5段階離散化
        #   < poor      : 0.0  (bad)
        #   < fair      : 0.25 (poor)
        #   < good      : 0.5  (fair)
        #   < excellent : 0.75 (good)
        #   >= excellent: 1.0  (excellent)
        # ----------------------------
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

        return {
            "contrast_raw": contrast_raw,
            "contrast_score": contrast_score,
            "contrast_grade": contrast_grade,
            "contrast_eval_status": "ok",
            "success": True,
        }
