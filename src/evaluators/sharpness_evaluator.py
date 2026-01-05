import cv2
import numpy as np
from typing import Dict, Any, Optional

from utils.image_utils import ImageUtils
from utils.app_logger import Logger


class SharpnessEvaluator:
    """
    画像のシャープネスを評価するクラス。

    出力:
        - sharpness_raw   : ラプラシアン分散の生値（高いほどシャープ）
        - sharpness_score : 0 / 0.25 / 0.5 / 0.75 / 1.0 の離散スコア
        - sharpness_eval_status : "ok" / "invalid_input" / "error"
    """

    RAW_KEY = "sharpness_raw"
    SCORE_KEY = "sharpness_score"

    def __init__(
        self,
        logger: Optional[Logger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            logger: ロガー。指定がない場合は内部で生成。
            config: 閾値設定など。
                例:
                sharpness:
                  discretize_thresholds_raw:
                    poor: 10.0
                    fair: 15.0
                    good: 22.0
                    excellent: 30.0
        """
        self.logger = logger or Logger(logger_name="SharpnessEvaluator")
        self.config = config or {}

        sharp_conf = self.config.get("sharpness", {})
        disc_conf = sharp_conf.get("discretize_thresholds_raw", {})

        # ラプラシアン分散（sharpness_raw）の閾値
        # ★ デフォルト値は現在のCSVの分布（だいたい 10〜35）からの仮置き。
        #    実写を見ながらあとで config から調整していく想定。
        self.threshold_poor: float = float(disc_conf.get("poor", 10.0))
        self.threshold_fair: float = float(disc_conf.get("fair", 15.0))
        self.threshold_good: float = float(disc_conf.get("good", 22.0))
        self.threshold_excellent: float = float(disc_conf.get("excellent", 30.0))

    def evaluate(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像のシャープネスを評価する。

        Args:
            image: 入力画像（BGR / RGB / グレースケール）

        Returns:
            dict:
                - sharpness_raw
                - sharpness_score
                - sharpness_eval_status
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            self.logger.warning("SharpnessEvaluator: invalid image. fallback to neutral.")
            return {
                self.RAW_KEY: None,
                self.SCORE_KEY: 0.5,   # ニュートラル
                "sharpness_eval_status": "invalid_input",
            }

        try:
            # カラーの場合はグレースケールに変換
            if image.ndim == 3:
                gray_image = ImageUtils.to_grayscale(image)
            else:
                gray_image = image

            # OpenCV の Laplacian が扱いやすいように uint8 に揃えておくと安定
            if gray_image.dtype != np.uint8:
                # 0〜255 にスケーリング
                g_min = float(gray_image.min())
                g_max = float(gray_image.max())
                if g_max > g_min:
                    norm = (gray_image.astype(np.float32) - g_min) / (g_max - g_min)
                    gray_image_u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    gray_image_u8 = np.zeros_like(gray_image, dtype=np.uint8)
            else:
                gray_image_u8 = gray_image

            # ラプラシアンを計算
            laplacian = cv2.Laplacian(gray_image_u8, cv2.CV_64F, ksize=3)

            # ラプラシアン分散 = シャープネス生値
            variance = float(laplacian.var())

            score = self._to_discrete_score(variance)

            result = {
                self.RAW_KEY: variance,
                self.SCORE_KEY: score,
                "sharpness_eval_status": "ok",
            }

            # 必要ならデバッグ用ログ
            self.logger.debug(
                f"sharpness_raw={variance:.3f}, sharpness_score={score:.2f}"
            )

            return result

        except Exception as e:
            self.logger.warning(f"SharpnessEvaluator: exception during evaluate: {e}")
            return {
                self.RAW_KEY: None,
                self.SCORE_KEY: 0.5,    # ニュートラル
                "sharpness_eval_status": "error",
            }

    # -----------------------
    # 内部ヘルパー
    # -----------------------

    def _to_discrete_score(self, raw: float) -> float:
        """
        ラプラシアン分散 raw を 0 / 0.25 / 0.5 / 0.75 / 1.0 にマッピングする。
        """
        v = float(raw)

        if v >= self.threshold_excellent:
            return 1.0
        if v >= self.threshold_good:
            return 0.75
        if v >= self.threshold_fair:
            return 0.5
        if v >= self.threshold_poor:
            return 0.25
        return 0.0
