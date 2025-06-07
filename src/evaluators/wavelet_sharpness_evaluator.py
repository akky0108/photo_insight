import numpy as np
import pywt
from utils.image_utils import ImageUtils


class WaveletSharpnessEvaluator:
    def evaluate(self, image: np.ndarray) -> float:
        """
        ウェーブレット変換を使用して画像のシャープネスを評価します。

        :param image: 入力画像（BGR形式またはグレースケール）
        :return: シャープネスのスコア（高いほどシャープネスが高い）
        """
        # 画像がカラーの場合はグレースケールに変換
        if len(image.shape) == 3:
            gray_image = ImageUtils.to_grayscale(image)
        else:
            gray_image = image

        # ウェーブレット変換を行う
        coeffs = pywt.wavedec2(gray_image, "haar", level=2)
        cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

        # シャープネスの尺度として、高周波成分の絶対平均値を使用
        sharpness_score = (
            np.mean(np.abs(cH1))
            + np.mean(np.abs(cV1))
            + np.mean(np.abs(cD1))
            + np.mean(np.abs(cH2))
            + np.mean(np.abs(cV2))
            + np.mean(np.abs(cD2))
        ) / 6

        return sharpness_score
