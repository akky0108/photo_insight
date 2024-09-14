import numpy as np

class MathUtils:
    
    @staticmethod
    def log_scale(value):
        """
        対数スケールに変換する。
        負の値に対応するため log(1 + value) を使用。
        :param value: 元の数値または数値配列
        :return: 対数スケールに変換された数値または配列
        """
        try:
            return np.log1p(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid input for log scale: {value}. Error: {e}")
    
    @staticmethod
    def min_max_normalization(value, min_value, max_value):
        """
        Min-Max正規化を行う。値を0～1の範囲にスケール。
        :param value: 正規化する元の値または配列
        :param min_value: 元データの最小値
        :param max_value: 元データの最大値
        :return: 0～1の範囲にスケールされた値または配列
        """
        if min_value == max_value:
            raise ValueError("min_value and max_value must be different.")
        
        normalized_value = (value - min_value) / (max_value - min_value)
        return np.clip(normalized_value, 0, 1)  # 0〜1の範囲に収める

    @staticmethod
    def z_score_normalization(value, mean, std):
        """
        Zスコア正規化を行う。標準偏差と平均を使用して値をスケール。
        :param value: 正規化する値または配列
        :param mean: 元データの平均
        :param std: 元データの標準偏差
        :return: Zスコアでスケールされた値または配列
        """
        if std == 0:
            raise ValueError("Standard deviation cannot be zero.")
        
        return (value - mean) / std

    @staticmethod
    def clip_values(value, min_value=0.0, max_value=1.0):
        """
        値を指定した範囲内にクリップする。
        :param value: クリップする値または配列
        :param min_value: クリップする最小値
        :param max_value: クリップする最大値
        :return: 指定した範囲内に収めた値または配列
        """
        return np.clip(value, min_value, max_value)

    @staticmethod
    def safe_divide(numerator, denominator, default_value=0.0):
        """
        ゼロ除算を回避し、安全に除算を行う。
        :param numerator: 分子
        :param denominator: 分母
        :param default_value: 分母がゼロの場合に返すデフォルト値
        :return: 安全に除算された結果
        """
        if denominator == 0:
            return default_value
        return numerator / denominator
