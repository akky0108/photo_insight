# -*- coding: utf-8 -*-
"""
rule_of_thirds_evaluator.py

ルール・オブ・サード（3分割法）に基づいて構図を評価するためのクラス。
画像と被写体中心座標（またはバウンディングボックス）からスコアを算出する。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class RuleOfThirdsEvaluator:
    """
    ルール・オブ・サード（3分割法）に基づいて画像構図を評価するクラス。

    ルール・オブ・サードとは、画像を縦横 3 等分することで重要な要素を配置するポイントを決め、
    よりバランスの取れた構図を作るための指標です。

    本クラスでは、「被写体の中心」が 3 分割のライン（または交点）にどれだけ近いかをスコア化します。
    スコアは 0.0〜1.0 の範囲に正規化され、1.0 に近いほど「サードにうまく乗っている」構図とみなします。
    """

    def __init__(self, image: np.ndarray):
        """
        コンストラクタ。評価対象の画像を受け取る。

        Parameters
        ----------
        image : np.ndarray
            評価対象の画像 (H, W, C) あるいは (H, W) の numpy 配列。
        """
        if image is None:
            raise ValueError("image is None")

        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be numpy.ndarray, got {type(image)}")

        if image.ndim not in (2, 3):
            raise ValueError(f"image must be 2D or 3D array, got ndim={image.ndim}")

        self.image = image
        self.height, self.width = image.shape[:2]

        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"image shape is invalid: height={self.height}, width={self.width}"
            )

        # あらかじめ 3 分割ラインを計算しておく
        self.thirds_x = (self.width / 3.0, 2.0 * self.width / 3.0)
        self.thirds_y = (self.height / 3.0, 2.0 * self.height / 3.0)

        # 正規化に用いる距離スケール
        # 「min(width, height) / 6」を 1.0 → 0.0 の変換範囲とする。
        self._norm_dist = min(self.width, self.height) / 6.0

    # ------------------------------------------------------------------
    # パブリック API
    # ------------------------------------------------------------------
    def evaluate(self) -> float:
        """
        画像中心を「被写体の中心」とみなして評価する。

        「顔中心や全身中心をまだ計算していない段階」や
        「単純に画像中心でのサード評価を見たい」場合の簡易版。

        Returns
        -------
        float
            サードスコア (0.0 ~ 1.0)。
        """
        center_point = (self.width / 2.0, self.height / 2.0)
        return self.evaluate_from_point(center_point)

    def evaluate_from_point(self, point: Tuple[float, float]) -> float:
        """
        任意の被写体中心点からサードスコアを計算する。

        典型的には以下のような値を渡すことを想定：
        - 顔検出の中心座標 (face_center)
        - 全身バウンディングボックスの中心
        - 「main_subject_center」のような抽象化された被写体中心

        Parameters
        ----------
        point : tuple(float, float)
            (x, y) 形式の画像座標（左上(0, 0)，右下(width, height)）。

        Returns
        -------
        float
            サードスコア (0.0 ~ 1.0)。
        """
        x, y = point

        # 画像内に収まっていない場合はクリップしておく（極端な外れ値を防ぐ）
        x = float(np.clip(x, 0.0, float(self.width)))
        y = float(np.clip(y, 0.0, float(self.height)))

        distance = self._nearest_thirds_distance(x, y)
        score = self._distance_to_score(distance)
        return score

    def evaluate_from_bbox(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> float:
        """
        バウンディングボックス（左上/右下）からサードスコアを計算する。

        bbox は (x_min, y_min, x_max, y_max) を想定。
        例えば、全身検出や顔検出の bbox をそのまま渡すことを想定。

        Parameters
        ----------
        bbox : tuple(float, float, float, float)
            (x_min, y_min, x_max, y_max) 形式の bbox。

        Returns
        -------
        float
            サードスコア (0.0 ~ 1.0)。
        """
        x_min, y_min, x_max, y_max = bbox

        # 座標が逆転している場合や、面積がない場合はフォールバックとして画像中心を使う
        if x_max <= x_min or y_max <= y_min:
            return self.evaluate()

        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return self.evaluate_from_point((cx, cy))

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------
    def _nearest_thirds_distance(self, x: float, y: float) -> float:
        """
        指定点 (x, y) から、「最も近いサードライン」までの距離を返す。

        水平方向（x）のサード 2 本、垂直方向（y）のサード 2 本のいずれかに対する
        最小距離を採用する。
        """
        tx1, tx2 = self.thirds_x
        ty1, ty2 = self.thirds_y

        dx = min(abs(x - tx1), abs(x - tx2))
        dy = min(abs(y - ty1), abs(y - ty2))

        # 「どちらか一方のラインに寄っていれば OK」とみなす設計のため、
        # dx と dy のうち小さい方を使用する。
        return min(dx, dy)

    def _distance_to_score(self, distance: float) -> float:
        """
        ラインまでの距離を 0.0〜1.0 のスコアに変換する。

        - distance = 0           → score = 1.0（サードライン/交点ど真ん中）
        - distance >= _norm_dist → score = 0.0（それ以上離れていても一律 0.0）

        Returns
        -------
        float
            スコア (0.0 ~ 1.0)。
        """
        if self._norm_dist <= 0:
            return 0.0

        score = 1.0 - (distance / self._norm_dist)
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        return float(score)
