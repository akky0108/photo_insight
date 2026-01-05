# evaluators/exposure_evaluator.py

import cv2
import numpy as np

class ExposureEvaluator:
    def __init__(self, lower: float = 0.14, upper: float = 0.55, margin: float = 0.10):
        """
        lower/upper: 0..1 の範囲
        暗めポートレート想定の初期値（あとでログ見て調整）
        """
        self.lower = lower
        self.upper = upper
        self.margin = margin

    def evaluate(self, image: np.ndarray) -> dict:
        # 注意: image は BGR 前提
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_f = gray.astype(np.float32)
        if np.issubdtype(gray.dtype, np.integer):
            gray_f /= float(np.iinfo(gray.dtype).max)
        else:
            mx = float(gray_f.max()) if gray_f.size else 0.0
            if mx > 1.5:      # 0..255系のfloat
                gray_f /= 255.0
            # 0..1系ならそのまま
        gray_f = np.clip(gray_f, 0.0, 1.0)

        mean = float(np.mean(gray_f))

        if self.lower <= mean <= self.upper:
            score = 1.0
        elif (self.lower - self.margin) <= mean <= (self.upper + self.margin):
            score = 0.5
        else:
            score = 0.0

        return {
            "exposure_score": score,
            "mean_brightness": mean,              # 0..1
            "mean_brightness_8bit": mean * 255.0  # デバッグ用
        }
