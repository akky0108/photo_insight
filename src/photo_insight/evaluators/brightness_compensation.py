# evaluators/brightness_compensation.py
from __future__ import annotations


def adjust_noise_by_brightness(noise_score: float, exposure_score: float) -> float:
    """
    A-2: ノイズスコアの明るさ依存補正（救済のみ）。
    - noise_score: 0〜1（高いほどノイズが少なく良い）
    - exposure_score: 0〜1

    方針（案A）:
    - 暗い画像だけ「救済」して少しだけスコアを押し上げる
    - 明るい画像ではスコアを変えない（減点しない）
    """
    score = float(noise_score)
    exp = float(exposure_score)

    # 暗い画像: ノイズに少し甘くする（+補正、下がらない）
    if exp < 0.4:
        t = (0.4 - exp) / 0.4  # 0〜1
        relax = 0.15 * t  # 最大 +0.15
        score = score + relax * (1.0 - score)

    return max(0.0, min(1.0, score))


def adjust_blur_by_brightness(blurriness_score: float, exposure_score: float) -> float:
    """
    B-2: ブレスコアの明るさ依存補正（救済のみ）。
    - blurriness_score: 0〜1（高いほどブレが少なく良い）
    - exposure_score: 0〜1

    方針（案A）:
    - 暗い画像だけ「救済」して少しだけスコアを押し上げる
    - 明るい画像ではスコアを変えない（減点しない）
    """
    score = float(blurriness_score)
    exp = float(exposure_score)

    # 暗い画像: ブレに少し甘い補正（+補正、下がらない）
    if exp < 0.4:
        t = (0.4 - exp) / 0.4  # 0〜1
        relax = 0.10 * t  # 最大 +0.10
        score = score + relax * (1.0 - score)

    return max(0.0, min(1.0, score))
