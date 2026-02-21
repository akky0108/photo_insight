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


def _collect_logged_messages(mock_logger) -> list[str]:
    """
    MagicMock logger の主要ログメソッドからメッセージ文字列を回収する。
    logger.error / logger.exception どちらでも拾えるようにする。
    """
    msgs: list[str] = []

    for method_name in ("error", "exception", "warning", "info", "debug"):
        method = getattr(mock_logger, method_name, None)
        if method is None:
            continue

        for call in getattr(method, "call_args_list", []):
            if not call.args:
                msgs.append("")
                continue
            msgs.append(str(call.args[0]))

    return msgs


class HookErrorProcessor(BaseBatchProcessor):
    def load_data(self):
        return []

    def _process_batch(self, batch):
        return []


def test_hook_exception_logged(tmp_path):
    """
    現状の HookManager は hook 例外を握りつぶしてログ出力し、実行自体は継続する想定。
    （= BaseBatch まで例外が伝播しないことがある）
    そのため、ここでは「例外が raise される」ではなく「ログに残る」を保証する。
    """
    config_path = _write_min_config(tmp_path)
    processor = HookErrorProcessor(
        config_path=config_path, logger=MagicMock(), max_workers=1
    )

    def failing_hook():
        raise RuntimeError("Hook failure!")

    processor.add_hook(HookType.PRE_SETUP, failing_hook)

    # 例外伝播は期待しない（HookManagerが握る可能性があるため）
    processor.execute()

    assert processor.logger.error.called or processor.logger.exception.called

    msgs = _collect_logged_messages(processor.logger)

    # ① hook種別が出ている（Frameworkとしての観測性）
    assert any("PRE_SETUP" in m for m in msgs), "PRE_SETUP に関するログが出力されていません"

    # ② 原因文言が出ている（実装依存なので弱めに）
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
    processor = CleanupErrorProcessor(
        config_path=config_path, logger=MagicMock(), max_workers=1
    )

    # cleanup は BaseBatch 側で fail_fast により raise される想定（ここは現状通っているはず）
    with pytest.raises(RuntimeError):
        processor.execute()

    assert processor.logger.error.called or processor.logger.exception.called

    msgs = _collect_logged_messages(processor.logger)
    assert any("Cleanup failed" in m for m in msgs), "Cleanup failed に関するログが出力されていません"
