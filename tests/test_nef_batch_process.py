import pytest
from unittest.mock import MagicMock
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from photo_insight.pipelines.nef.nef_batch_process import NEFFileBatchProcess


# --- 共通フィクスチャ ---
@pytest.fixture
def dummy_processor(tmp_path):
    processor = NEFFileBatchProcess(config_path=None)
    processor.project_root = tmp_path
    processor.temp_dir = tmp_path / "temp"
    processor.temp_dir.mkdir(parents=True, exist_ok=True)
    processor.logger = MagicMock()
    return processor


def test_get_data_calls_load_data_and_is_cached(monkeypatch, dummy_processor):
    calls = {"n": 0}

    def fake_load_data(*args, **kwargs):
        calls["n"] += 1
        return [{"path": "/mock/a.NEF"}, {"path": "/mock/b.NEF"}]

    # NOTE: 現行Base契約では get_data() -> load_data()
    monkeypatch.setattr(dummy_processor, "load_data", fake_load_data)

    r1 = dummy_processor.get_data()
    r2 = dummy_processor.get_data()  # cache hit

    assert len(r1) == 2
    assert r1 == r2
    assert calls["n"] == 1  # 2回呼んでも load_data は1回


def test_process_aggregates_results_into_summary(monkeypatch, dummy_processor):
    # data は Base.process が参照するので、load_data を差し替えて供給
    monkeypatch.setattr(
        dummy_processor,
        "load_data",
        lambda *a, **k: [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
    )

    # batch を2分割させたいので max_workers/ batch_size を明示
    dummy_processor.max_workers = 2
    dummy_processor.batch_size = 2  # _generate_batches が参照

    # _process_batch の戻りは List[Dict]（status/score を見て summary が作られる）
    def fake_process_batch(batch):
        out = []
        for item in batch:
            out.append({"status": "success", "score": 0.5, "id": item["id"]})
        return out

    monkeypatch.setattr(dummy_processor, "_process_batch", fake_process_batch)

    # Base.process を実行
    dummy_processor.process()

    assert dummy_processor.final_summary["total"] == 4
    assert dummy_processor.final_summary["success"] == 4
    assert dummy_processor.final_summary["failure"] == 0
    assert dummy_processor.final_summary["avg_score"] == 0.5
    assert len(dummy_processor.all_results) == 4


def test_process_batch_empty_batch_ok(monkeypatch, dummy_processor):
    # 空バッチでも例外が出ない（NEF実装に依存しない形で担保）
    monkeypatch.setattr(dummy_processor, "_process_batch", lambda batch: [])
    dummy_processor.max_workers = 1
    dummy_processor.batch_size = 10

    # load_data も空にする
    monkeypatch.setattr(dummy_processor, "load_data", lambda *a, **k: [])
    # process() は "No data to process" で return する想定
    dummy_processor.process()


def test_execute_calls_process(monkeypatch, dummy_processor):
    # Base.execute は kwargs を process に渡すので、テスト側は **kwargs を受ける
    called = {}

    def fake_process(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs

    monkeypatch.setattr(dummy_processor, "process", fake_process)

    dummy_processor.execute(target_dir=Path("/mock/target"))

    assert "kwargs" in called
    assert called["kwargs"]["target_dir"] == Path("/mock/target")


def test_thread_safe_output_data():
    processor = NEFFileBatchProcess()
    processor.output_data = []

    def add_data():
        for _ in range(100):
            with processor.get_lock():
                processor.output_data.append({"value": 1})

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(add_data) for _ in range(10)]
        for future in futures:
            future.result()  # 例外チェック用

    assert len(processor.output_data) == 1000
