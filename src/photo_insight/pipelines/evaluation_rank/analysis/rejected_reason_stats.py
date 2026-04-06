# src/batch_processor/evaluation_rank/analysis/rejected_reason_stats.py
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _to_bool(v: Any) -> bool:
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


def _first_nonempty_str(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


# -------------------------
# reason_code extraction
# -------------------------

_CODE_SAFE_RE = re.compile(r"[^a-z0-9_]+")


def _sanitize_code(s: str) -> str:
    """
    集計キーを安定化:
    - lower
    - spaces -> _
    - 記号は _ に寄せる（英数字 + _ のみにする）
    """
    s = (s or "").strip().lower()
    s = s.replace(" ", "_")
    s = _CODE_SAFE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def extract_reason_code(raw: Optional[str]) -> str:
    """
    raw reason text から reason_code を抽出する。

    例:
      "EYE_HALF_NG(p=0.9,sz=80)" -> "eye_half_ng"
      "EYE_CLOSED_WARN(p=0.99,sz=120)" -> "eye_closed_warn"
      "SEC:portrait group=A-1 overall=..." -> "sec"
      "portrait ... | FILL_RELAX" -> "portrait_..."（=先頭トークン）
    """
    if raw is None:
        return "unknown"

    s = str(raw).strip()
    if not s:
        return "unknown"

    s_low = s.strip()

    # 1) "AAA | BBB" 形式なら先頭を採用（EYE_* や base reason を安定的に拾う）
    head = s_low.split("|", 1)[0].strip()

    # 2) "SEC:portrait ..." 形式なら ":" の前を採用
    if ":" in head:
        head = head.split(":", 1)[0].strip()

    # 3) "EYE_HALF_NG(p=...)" のように "(" があれば前を採用
    if "(" in head:
        head = head.split("(", 1)[0].strip()

    code = _sanitize_code(head)
    return code or "unknown"


@dataclass(frozen=True)
class RejectedReasonSummaryRow:
    reason_code: str
    count: int
    ratio: float


class RejectedReasonAnalyzer:
    """
    Phase2:
    - accepted_flag == False の行だけ対象
    - (rejected_reason / accepted_reason) から reason_code を抽出して集計
    """

    def __init__(
        self,
        *,
        alias_map: Optional[Mapping[str, str]] = None,
        unknown_label: str = "unknown",
        # ★ 理由列の候補（順番に探す）
        reason_keys: Sequence[str] = ("rejected_reason", "accepted_reason"),
        # ★ raw を未知扱いに寄せるか（基本は False 推奨）
        keep_unknown_raw: bool = False,
    ) -> None:
        self.alias_map = {str(k).lower(): str(v) for k, v in (alias_map or {}).items()}
        self.unknown_label = str(unknown_label)
        self.reason_keys = tuple(reason_keys)
        self.keep_unknown_raw = bool(keep_unknown_raw)

    def normalize_code(self, code: str) -> str:
        code = _sanitize_code(code)
        if not code or code in ("nan", "none", "null"):
            return self.unknown_label

        mapped = self.alias_map.get(code)
        if mapped:
            return _sanitize_code(mapped) or self.unknown_label

        if code == self.unknown_label:
            return self.unknown_label

        # Phase2は基本 code をそのまま採用
        return code

    def analyze(self, rows: Iterable[Dict[str, Any]]) -> Tuple[List[RejectedReasonSummaryRow], Dict[str, Any]]:
        counter: Counter[str] = Counter()
        total_rows = 0
        total_rejected = 0
        missing_reason_rows = 0

        for r in rows:
            total_rows += 1
            if _to_bool(r.get("accepted_flag", False)):
                continue

            total_rejected += 1

            raw = _first_nonempty_str(r, self.reason_keys)
            if raw is None:
                missing_reason_rows += 1

            code = extract_reason_code(raw)
            code = self.normalize_code(code)

            # keep_unknown_raw=True のときだけ unknown に raw を付与（デバッグ用）
            if self.keep_unknown_raw and code == self.unknown_label and raw:
                code = f"{self.unknown_label}:{_sanitize_code(raw) or 'raw'}"

            counter[code] += 1

        summary: List[RejectedReasonSummaryRow] = []
        for code, count in counter.items():
            ratio = (count / total_rejected) if total_rejected > 0 else 0.0
            summary.append(RejectedReasonSummaryRow(reason_code=str(code), count=int(count), ratio=float(ratio)))

        summary.sort(key=lambda x: (-x.count, x.reason_code))

        meta = {
            "total_rows": total_rows,
            "total_rejected": total_rejected,
            "unique_reasons": len(counter),
            "missing_reason_rows": missing_reason_rows,
            "reason_keys": list(self.reason_keys),
        }
        return summary, meta


def write_rejected_reason_summary_csv(
    summary: List[RejectedReasonSummaryRow],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Phase2: reason_code を出す
    lines = ["reason_code,count,ratio\n"]
    for row in summary:
        lines.append(f"{row.reason_code},{row.count},{row.ratio:.6f}\n")
    out_path.write_text("".join(lines), encoding="utf-8")
