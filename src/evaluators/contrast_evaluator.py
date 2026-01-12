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

    def __init__(
        self,
        raw_floor: float = 5.0,   # これ未満は「ほぼ真っ平 → 低コントラスト」
        raw_ceil: float = 50.0,   # これ以上は「十分にコントラストあり」とみなして頭打ち
        gamma: float = 0.9,       # mid〜high を少し持ち上げる補正
    ) -> None:
        self.raw_floor = float(raw_floor)
        self.raw_ceil = float(raw_ceil)
        self.gamma = float(gamma)

    def evaluate(self, image: np.ndarray) -> dict:
        if not isinstance(image, np.ndarray):
            raise ValueError(
                "Invalid input: expected a numpy array representing an image."
            )

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
        # raw → [0, 1] の連続値に正規化
        # ----------------------------
        #   raw_floor 未満: 0.0
        #   raw_ceil 以上: 1.0
        #   その間は線形補間＋ガンマ補正
        rf = self.raw_floor
        rc = self.raw_ceil

        if rc <= rf:
            # パラメータが変でも一応動くように
            norm = 1.0
        else:
            norm = (contrast_raw - rf) / (rc - rf)
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0

        # mid〜high を少し持ち上げる
        # （暗部〜低コントラストはシビア、高コントラストは飽和しにくくする）
        norm = float(norm ** self.gamma)

        # ----------------------------
        # 5段階スコア + grade
        #  (NoiseEvaluator のスケールポリシーに合わせる)
        # ----------------------------
        if norm >= 0.85:
            contrast_score = 1.0
            contrast_grade = "excellent"
        elif norm >= 0.70:
            contrast_score = 0.75
            contrast_grade = "good"
        elif norm >= 0.50:
            contrast_score = 0.5
            contrast_grade = "fair"
        elif norm >= 0.30:
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
