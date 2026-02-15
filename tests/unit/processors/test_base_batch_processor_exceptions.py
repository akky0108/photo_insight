import yaml
import pytest
from unittest.mock import MagicMock

from photo_insight.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.batch_framework.core.hook_manager import HookType


def _write_min_config(tmp_path) -> str:
    cfg = {
        "batch": {"memory_threshold": 90, "fail_fast": True},
        "debug": {
            "log_summary_detail": False,
            # persist_run_results がデフォルト True ならテスト中にファイルI/Oが走る可能性があるのでOFF推奨
            "persist_run_results": False,
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)

def _logged_messages(mock_logger) -> list[str]:
    msgs = []
    for call in mock_logger.error.call_args_list:
        if call.args:
            msgs.append(str(call.args[0]))
        else:
            msgs.append("")
    return msgs


class HookErrorProcessor(BaseBatchProcessor):
    def load_data(self):
        return []

    def _process_batch(self, batch):
        return []


def test_hook_exception_logged(tmp_path):
    config_path = _write_min_config(tmp_path)
    processor = HookErrorProcessor(config_path=config_path, logger=MagicMock(), max_workers=1)

    def failing_hook():
        raise RuntimeError("Hook failure!")

    processor.add_hook(HookType.PRE_SETUP, failing_hook)

    with pytest.raises(RuntimeError):
        processor.execute()

    assert processor.logger.error.called
    msgs = _logged_messages(processor.logger)

    # ① hook種別が出ている（Frameworkとしての観測性）
    assert any("PRE_SETUP" in m for m in msgs), "PRE_SETUP に関するログが出力されていません"

    # ② 可能なら原因文言も出ている（実装依存なので弱めに）
    assert any("Hook failure" in m for m in msgs), "Hook failure に関するログが出力されていません"


class CleanupErrorProcessor(BaseBatchProcessor):
    def load_data(self):
        return []

    def _process_batch(self, batch):
        return []

    def cleanup(self):
        raise RuntimeError("Cleanup failed!")


def test_cleanup_exception_logged(tmp_path):
    config_path = _write_min_config(tmp_path)
    processor = CleanupErrorProcessor(config_path=config_path, logger=MagicMock(), max_workers=1)

    with pytest.raises(RuntimeError):
        processor.execute()

    assert processor.logger.error.called
    assert any(
        "Cleanup failed" in ((call.args[0] if call.args else ""))
        for call in processor.logger.error.call_args_list
    )
