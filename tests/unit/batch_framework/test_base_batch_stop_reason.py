# tests/unit/batch_framework/test_base_batch_stop_reason.py
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor


class DummyProcessor(BaseBatchProcessor):
    """
    BaseBatch の summary.json 生成（stop_reason含む）を検証するための最小実装。
    - load_data: 3件返す（= 3バッチになりやすい）
    - _process_batch:
        - 1バッチ目で memory_threshold_exceeded を立てる
        - 以降バッチは stop 中として空配列を返す（結果数が増えないようにする）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # batch_size を強制的に小さくして batch 分割を安定させる
        self.batch_size = 1
        self._call_n = 0

    def load_data(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return [{"id": 1}, {"id": 2}, {"id": 3}]

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._call_n += 1

        # 1回目のバッチで停止フラグを立てる（= memory stop を再現）
        if self._call_n == 1:
            self.memory_threshold_exceeded = True
            return [{"status": "success", "score": 0.5, "id": batch[0]["id"]}]

        # stop 中は何も返さない（Base の集計に入らない）
        if getattr(self, "memory_threshold_exceeded", False):
            return []

        return [{"status": "success", "score": 0.5, "id": batch[0]["id"]}]


@pytest.fixture
def processor():
    # --- ConfigManager モック（BaseBatch が参照する最小を用意） ---
    config_manager = MagicMock()
    config_manager.config = {
        "debug": {
            "persist_run_results": True,  # ★summary.json を吐かせる
            "run_results_dir": "runs",
        },
        "batch": {
            "fail_fast": True,
            "cleanup_fail_fast": False,
        },
    }
    config_manager.get_logger.return_value = MagicMock()
    config_manager.reload_config = MagicMock()

    # --- HookManager / SignalHandler は no-op でOK ---
    hook_manager = MagicMock()
    hook_manager.execute_hooks = MagicMock()
    hook_manager.add_hook = MagicMock()
    hook_manager.logger = MagicMock()

    signal_handler = MagicMock()
    signal_handler.logger = MagicMock()

    p = DummyProcessor(
        config_path=None,
        max_workers=2,
        config_manager=config_manager,
        hook_manager=hook_manager,
        signal_handler=signal_handler,
        logger=MagicMock(),
    )

    # --- ResultStore をモックして summary.json の中身を捕まえる ---
    p.result_store = MagicMock()

    run_ctx = MagicMock()
    run_ctx.out_dir = "dummy_out_dir"

    p.result_store.make_run_context.return_value = run_ctx

    # save_jsonl は呼ばれても良い（ここでは内容は見ない）
    p.result_store.save_jsonl = MagicMock()

    # save_json に渡ってくる obj を検査したい
    captured: Dict[str, Any] = {}

    def _capture_save_json(_run_ctx, obj: Dict[str, Any], name: str):
        # name が summary.json のときだけ拾う
        if name == "summary.json":
            captured.clear()
            captured.update(obj)

    p.result_store.save_json.side_effect = _capture_save_json

    # finalize は呼ばれても良い
    p.result_store.finalize_to_final_dir.return_value = None

    return p, captured


def test_summary_json_includes_stop_reason_memory_threshold(processor):
    p, captured = processor

    # execute() を回すと BaseBatch が summary.json を save_json() する想定
    p.execute()

    # summary.json が保存されたこと（= captured が埋まってること）
    assert captured, "summary.json payload was not captured (save_json not called?)"

    # ★ここが本題：stop_reason が入ること（memory_threshold を想定）
    # ※ BaseBatch 側の実装に合わせて文字列は統一してください
    assert captured.get("stop_reason") == "memory_threshold"

    # interrupted / completed_all_batches の整合
    assert captured.get("interrupted") is True
    assert captured.get("completed_all_batches") is False

    # 結果件数も “実処理数” と合う（1バッチ目だけ成功を返す設計）
    assert captured.get("total") == 1
    assert captured.get("success") == 1
    assert captured.get("failure") == 0
