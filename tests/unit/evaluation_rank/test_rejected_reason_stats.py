# tests/unit/evaluation_rank/test_rejected_reason_stats.py

from __future__ import annotations

from pathlib import Path

from photo_insight.batch_processor.evaluation_rank.analysis.rejected_reason_stats import (
    RejectedReasonAnalyzer,
    extract_reason_code,
    write_rejected_reason_summary_csv,
)


def test_extract_reason_code_eye_half_ng():
    raw = "EYE_HALF_NG(p=0.91,sz=120)"
    assert extract_reason_code(raw) == "eye_half_ng"


def test_extract_reason_code_eye_closed_warn():
    raw = "EYE_CLOSED_WARN(p=0.99,sz=80)"
    assert extract_reason_code(raw) == "eye_closed_warn"


def test_extract_reason_code_sec_prefix():
    raw = "SEC:portrait group=A-1 overall=77.71 f=70.0 c=60.0"
    assert extract_reason_code(raw) == "sec"


def test_extract_reason_code_pipe_takes_head():
    raw = "portrait group=a-1 st=upper_body rank=1/4 o=88.0 | FILL_RELAX"
    assert extract_reason_code(raw).startswith("portrait_group_a_1")


def test_rejected_reason_analyzer_counts_by_reason_code():
    rows = [
        {"accepted_flag": 0, "accepted_reason": "SEC:portrait group=A-1 overall=70.0"},
        {"accepted_flag": 0, "accepted_reason": "SEC:portrait group=A-2 overall=71.0"},
        {"accepted_flag": 0, "accepted_reason": "EYE_HALF_NG(p=0.91,sz=120)"},
        {"accepted_flag": 1, "accepted_reason": "ignored"},  # accepted は対象外
        {"accepted_flag": 0, "rejected_reason": ""},  # unknown
    ]

    analyzer = RejectedReasonAnalyzer()
    summary, meta = analyzer.analyze(rows)

    m = {s.reason_code: s.count for s in summary}
    assert m["sec"] == 2
    assert m["eye_half_ng"] == 1
    assert m["unknown"] == 1

    assert meta["total_rows"] == 5
    assert meta["total_rejected"] == 4


def test_write_rejected_reason_summary_csv_writes_reason_code_header(tmp_path: Path):
    analyzer = RejectedReasonAnalyzer()
    summary, _ = analyzer.analyze([{"accepted_flag": 0, "accepted_reason": "SEC:portrait ..."}])

    out = tmp_path / "rejected_reason_summary.csv"
    write_rejected_reason_summary_csv(summary, out)

    text = out.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "reason_code,count,ratio"
