from platform import processor
import yaml
import pytest
from unittest.mock import MagicMock

from batch_framework.base_batch import BaseBatchProcessor
from batch_framework.core.hook_manager import HookType


def _write_min_config(tmp_path) -> str:
    cfg = {
        "batch": {"memory_threshold": 90},
        "debug": {"log_summary_detail": False},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)


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

    # 現行の Base はフック例外で execute() 自体は落とさない（ログに残す）
    processor.execute()

    assert processor.logger.error.called
    assert any(
        "Hook failure" in str(call.args)
        for call in processor.logger.error.call_args_list
    ), "Hook failure に関するログが出力されていません"


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

    # 現行の Base は cleanup 例外を errors に積み、最後に handle_error(..., raise_exception=True) で raise する
    with pytest.raises(RuntimeError):
        processor.execute()

    assert processor.logger.error.called
    assert any(
        "Cleanup failed" in str(call.args)
        for call in processor.logger.error.call_args_list
    )
