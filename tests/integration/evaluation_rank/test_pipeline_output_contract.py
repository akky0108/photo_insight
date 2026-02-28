from __future__ import annotations

import csv
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_pipeline_produces_ranking_csv_with_contract_header(
    tmp_path: Path, minimal_required_rows, required_columns, monkeypatch
):
    from photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor import (
        EvaluationRankBatchProcessor,
    )
    from photo_insight.batch_processor.evaluation_rank.contract import OUTPUT_COLUMNS

    eval_dir = tmp_path / "temp"
    out_dir = tmp_path / "output"
    eval_dir.mkdir()
    out_dir.mkdir()

    date = "2099-01-01"
    input_csv = eval_dir / f"evaluation_results_{date}.csv"

    with input_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=required_columns)
        w.writeheader()
        for r in minimal_required_rows:
            w.writerow({k: r.get(k, "") for k in required_columns})

    # scorer / acceptance を軽量化（writerは本物）
    class DummyScorer:
        def build_calibration(self, data):  # noqa
            self.calibration = {}

        def technical_score(self, row):  # noqa
            return 0.1

        def face_score(self, row):  # noqa
            return 0.2

        def composition_score(self, row):  # noqa
            return 0.3

        def overall_score(self, row, tech_sp, face_sp, comp_sp):  # noqa
            return 0.42

    class DummyAcceptance:
        def run(self, rows):  # noqa
            return {"portrait": 0.0, "non_face": 0.0}

    monkeypatch.setattr(
        "photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor.EvaluationScorer", DummyScorer
    )
    monkeypatch.setattr(
        "photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor.AcceptanceEngine",
        DummyAcceptance,
    )

    # 周辺副作用を潰して安定化
    monkeypatch.setattr(
        "photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor.apply_lightroom_fields",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor.EvaluationRankBatchProcessor._write_provisional_vs_accepted_summary",
        lambda self, rows: None,
    )
    monkeypatch.setattr(
        "photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor.EvaluationRankBatchProcessor._write_rejected_reason_summary",
        lambda self, rows: None,
    )
    monkeypatch.setattr(
        "photo_insight.batch_processor.evaluation_rank.evaluation_rank_batch_processor.EvaluationRankBatchProcessor._apply_provisional_top_percent",
        lambda self, rows: None,
    )

    p = EvaluationRankBatchProcessor(config_path=None, max_workers=1, date=date)

    # setup/config に依存しないよう tests から paths を直指定
    p.paths["evaluation_data_dir"] = str(eval_dir)
    p.paths["output_data_dir"] = str(out_dir)

    data = p.load_data()
    p.after_data_loaded(data)
    p.all_results = p._process_batch(data)
    p.cleanup()

    out_csv = out_dir / f"evaluation_ranking_{date}.csv"
    assert out_csv.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == list(OUTPUT_COLUMNS)
