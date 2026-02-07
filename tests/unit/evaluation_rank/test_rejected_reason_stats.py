# tests/unit/evaluation_rank/test_rejected_reason_stats.py
from __future__ import annotations

from pathlib import Path

from batch_processor.evaluation_rank.analysis.rejected_reason_stats import (
    RejectedReasonAnalyzer,
    write_rejected_reason_summary_csv,
)


def test_rejected_reason_analyze_counts_only_rejected():
    rows = [
        {"file_name": "a", "accepted_flag": 1, "rejected_reason": "blur_low"},
        {"file_name": "b", "accepted_flag": 0, "rejected_reason": "blur_low"},
        {"file_name": "c", "accepted_flag": False, "rejected_reason": "noise_high"},
        {"file_name": "d", "accepted_flag": "0", "rejected_reason": None},
        {"file_name": "e", "accepted_flag": "true", "rejected_reason": "blur_low"},
    ]
    analyzer = RejectedReasonAnalyzer(alias_map={"blurry": "blur_low"})
    summary, meta = analyzer.analyze(rows)

    assert meta["total_rows"] == 5
    assert meta["total_rejected"] == 3  # b,c,d
    # blur_low:1, noise_high:1, unknown:1
    got = {r.reason: r.count for r in summary}
    assert got["blur_low"] == 1
    assert got["noise_high"] == 1
    assert got["unknown"] == 1


def test_write_rejected_reason_summary_csv(tmp_path: Path):
    analyzer = RejectedReasonAnalyzer()
    summary, _ = analyzer.analyze(
        [
            {"accepted_flag": 0, "rejected_reason": "blur_low"},
            {"accepted_flag": 0, "rejected_reason": "blur_low"},
            {"accepted_flag": 0, "rejected_reason": "noise_high"},
        ]
    )
    out = tmp_path / "rejected_reason_summary.csv"
    write_rejected_reason_summary_csv(summary, out)

    text = out.read_text(encoding="utf-8")
    assert "reason,count,ratio" in text
    assert "blur_low,2," in text
    assert "noise_high,1," in text
