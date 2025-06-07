# evaluators/exposure_evaluator.py

import cv2
import numpy as np


class ExposureEvaluator:
    def __init__(self, lower_thresh: float = 90, upper_thresh: float = 160):
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh

    def evaluate(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if self.lower_thresh <= mean_brightness <= self.upper_thresh:
            score = 1.0
        elif (self.lower_thresh - 20) <= mean_brightness <= (self.upper_thresh + 20):
            score = 0.5
        else:
            score = 0.0

        return {"exposure_score": score, "mean_brightness": mean_brightness}
