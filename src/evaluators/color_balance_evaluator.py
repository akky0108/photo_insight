import numpy as np
from typing import Dict


class ColorBalanceEvaluator:
    def evaluate(self, image: np.ndarray) -> Dict[str, float]:
        """
        ホワイトバランス指標を評価し、スコアを返す。
        :param image: BGR形式の画像（OpenCV互換）
        :return: スコア辞書 {"white_balance_score": float}
        """
        # 仮スコア計算：平均のRGB値の標準偏差を使った簡易評価
        mean = image.mean(axis=(0, 1))  # BGR平均
        std = np.std(mean)
        score = max(0.0, 1.0 - std / 128.0)  # 値は0〜1に正規化（仮）

        return {"white_balance_score": round(score, 4)}
