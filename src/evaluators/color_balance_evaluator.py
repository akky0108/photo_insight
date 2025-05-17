import numpy as np
from typing import Dict


class ColorBalanceEvaluator:
    def __init__(self):
        # 基本的に初期化は不要
        pass

    def evaluate(self, image: np.ndarray) -> Dict[str, float]:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid numpy.ndarray.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be an RGB image with 3 channels.")

        # ホワイトバランス評価（グレーワールド仮説）
        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])
        rgb_mean = np.mean([r_mean, g_mean, b_mean])

        white_balance_score = 1.0 - (np.std([r_mean, g_mean, b_mean]) / rgb_mean)
        white_balance_score = np.clip(white_balance_score, 0.0, 1.0)

        # 肌色スコア評価（例：肌色範囲にあるピクセルの割合）
        skin_mask = self._get_skin_mask(image)
        skin_tone_score = np.sum(skin_mask) / (image.shape[0] * image.shape[1])
        skin_tone_score = np.clip(skin_tone_score, 0.0, 1.0)

        # 総合スコア（シンプルに平均）
        color_balance_score = np.mean([white_balance_score, skin_tone_score])

        return {
            "color_balance_score": color_balance_score,
            "white_balance_score": white_balance_score,
            "skin_tone_score": skin_tone_score
        }

    def _get_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """肌色に近いピクセルを判定する簡易マスク（RGBベース）"""
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        b = image[:, :, 2].astype(np.float32)

        condition1 = (r > 95) & (g > 40) & (b > 20)
        condition2 = ((np.max(image, axis=2) - np.min(image, axis=2)) > 15)
        condition3 = (np.abs(r - g) > 15)
        condition4 = (r > g) & (r > b)

        skin_mask = condition1 & condition2 & condition3 & condition4
        return skin_mask.astype(np.uint8)
