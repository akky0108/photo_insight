from abc import ABC, abstractmethod
import numpy as np


class BaseCompositionEvaluator(ABC):
    @abstractmethod
    def evaluate(self, image: np.ndarray, face_boxes: list) -> dict:
        """
        画像と顔検出結果（face_boxes）を元に構図評価を行い、その結果を辞書形式で返す。
        """
        pass
