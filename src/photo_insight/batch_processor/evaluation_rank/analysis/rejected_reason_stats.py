# src/batch_processor/evaluation_rank/analysis/rejected_reason_stats.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _to_bool(v: Any) -> bool:
    # accepted_flag が 0/1, "0"/"1", bool などでも壊れないようにする
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "t", "yes", "y"):
            return True
        if s in ("0", "false", "f", "no", "n", ""):
            return False
    return bool(v)


@dataclass(frozen=True)
class RejectedReasonSummaryRow:
    reason: str
    count: int
    ratio: float


class RejectedReasonAnalyzer:
    """
    Phase1:
    - accepted_flag == False の行だけ対象
    - rejected_reason を正規化
    - reasonごとの件数と比率を出す
    """

    def __init__(
        self,
        *,
        alias_map: Optional[Mapping[str, str]] = None,
        unknown_label: str = "unknown",
        keep_unknown_raw: bool = False,  # Phase1は基本False推奨（unknownに寄せる）
    ) -> None:
        self.alias_map = {k.lower(): v for k, v in (alias_map or {}).items()}
        self.unknown_label = unknown_label
        self.keep_unknown_raw = keep_unknown_raw

    def normalize_reason(self, reason: Any) -> str:
        if reason is None:
            return self.unknown_label
        s = str(reason).strip()
        if not s:
            return self.unknown_label

        key = s.lower()
        if key in ("nan", "none", "null"):
            return self.unknown_label

        # alias (揺れ) 吸収
        mapped = self.alias_map.get(key)
        if mapped:
            return mapped

        # 最低限の正規化（空白→アンダースコア、英数字以外は維持）
        key2 = key.replace(" ", "_")
        if not key2:
            return self.unknown_label

        # 未知の扱い（Phase1は unknown に寄せるのが運用向き）
        if self.keep_unknown_raw:
            return f"{self.unknown_label}:{key2}"
        return key2

    def analyze(self, rows: Iterable[Dict[str, Any]]) -> Tuple[List[RejectedReasonSummaryRow], Dict[str, Any]]:
        counter: Counter[str] = Counter()
        total_rows = 0
        total_rejected = 0

        for r in rows:
            total_rows += 1
            accepted = _to_bool(r.get("accepted_flag", False))
            if accepted:
                continue
            total_rejected += 1
            reason = self.normalize_reason(r.get("rejected_reason"))
            counter[reason] += 1

        summary: List[RejectedReasonSummaryRow] = []
        for reason, count in counter.items():
            ratio = (count / total_rejected) if total_rejected > 0 else 0.0
            summary.append(RejectedReasonSummaryRow(reason=reason, count=int(count), ratio=float(ratio)))

        summary.sort(key=lambda x: (-x.count, x.reason))

        meta = {
            "total_rows": total_rows,
            "total_rejected": total_rejected,
            "unique_reasons": len(counter),
        }
        return summary, meta


def write_rejected_reason_summary_csv(
    summary: List[RejectedReasonSummaryRow],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["reason,count,ratio\n"]
    for row in summary:
        lines.append(f"{row.reason},{row.count},{row.ratio:.6f}\n")
    out_path.write_text("".join(lines), encoding="utf-8")
