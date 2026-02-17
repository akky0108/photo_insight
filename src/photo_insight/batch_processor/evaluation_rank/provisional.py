# batch_processor/evaluation_rank/provisional.py
from __future__ import annotations

import math
from typing import Any, Dict, List


def apply_provisional_top_percent(
    records: List[Dict[str, Any]],
    percent: float,
    score_key: str = "overall_score",
    flag_key: str = "provisional_top_percent_flag",
    percent_key: str = "provisional_top_percent",
) -> None:
    """
    records に「上位 percent%」の暫定フラグを付与する（in-place）。
    - accepted 判定とは独立
    - 同点は index 方式（単純にソート後の上位 k 件）で扱う

    注意:
      - records の順序は変更しない（内部でソートして付与対象を決めるだけ）
      - score が None/欠損のものは最下位扱い
    """
    # sanitize
    if percent is None:
        percent = 0.0
    try:
        p = float(percent)
    except Exception:
        p = 0.0
    p = max(0.0, min(100.0, p))

    n = len(records)
    if n == 0:
        return

    # k (ceil) - 例: 12件で10% -> 2件
    k = int(math.ceil(n * (p / 100.0)))

    # 全行に percent をセット（分析・可視化で便利）
    for r in records:
        r[percent_key] = p
        r[flag_key] = 0

    if k <= 0:
        return

    # 元順序を壊さないため、index を保持したまま並べ替え
    def score_val(rec: Dict[str, Any]) -> float:
        v = rec.get(score_key, None)
        if v is None:
            return float("-inf")
        # numpy / float32 などでも落ちないように float 化を試みる
        try:
            return float(v)
        except Exception:
            return float("-inf")

    indexed = list(enumerate(records))
    indexed.sort(key=lambda t: score_val(t[1]), reverse=True)

    top_indices = {idx for idx, _ in indexed[:k]}
    for i in top_indices:
        records[i][flag_key] = 1
