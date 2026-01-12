# evaluators/brightness_compensation.py
from __future__ import annotations

def adjust_noise_by_brightness(noise_score: float, exposure_score: float) -> float:
    """
    A-2: ノイズスコアの明るさ依存補正。
    noise_score: 0〜1（高いほどノイズが少なく良い）
    exposure_score: 0〜1
    """
    score = float(noise_score)
    exp = float(exposure_score)

    # 暗い画像: ノイズに少し甘くする（+補正）
    if exp < 0.4:
        t = (0.4 - exp) / 0.4          # 0〜1
        relax = 0.15 * t              # 最大 +0.15
        score = score + relax * (0.5 - score)

    # 明るい画像: ノイズに少し厳しくする（-補正）
    elif exp > 0.7:
        t = (exp - 0.7) / 0.3          # 0〜1
        strict = 0.15 * t              # 最大 -0.15
        score = score * (1.0 - strict)

    return max(0.0, min(1.0, score))


def adjust_blur_by_brightness(blurriness_score: float, exposure_score: float) -> float:
    """
    B-2: ブレスコアの明るさ依存補正。
    blurriness_score: 0〜1（高いほどブレが少なく良い）
    exposure_score: 0〜1
    """
    score = float(blurriness_score)
    exp = float(exposure_score)

    # 暗い画像: ブレに少し甘い補正（+補正）
    if exp < 0.4:
        t = (0.4 - exp) / 0.4          # 0〜1
        relax = 0.10 * t              # 最大 +0.10
        score = score + relax * (1.0 - score)

    # 明るい画像: ブレに少し厳しい補正（-補正）
    elif exp > 0.7:
        t = (exp - 0.7) / 0.3          # 0〜1
        strict = 0.10 * t              # 最大 -0.10
        score = score - strict * (score - 0.5)

    return max(0.0, min(1.0, score))
