# tests/integration/test_base_batch_processor.py

import pytest
import logging
import signal
import json
from unittest.mock import  Mock, call, MagicMock
from batch_framework.core.hook_manager import HookType
from batch_framework.core.signal_handler import SignalHandler
from batch_framework.core.config_manager import ConfigManager
from tests.integration.dummy_batch_processor import DummyBatchProcessor

def test_execute_runs_all_hooks_and_methods(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))

    config_manager = ConfigManager(config_path=str(config_path))

    # execute_hooks に差し替え
    mock_hook_manager = Mock()
    mock_hook_manager.execute_hooks = Mock()

    mock_logger = Mock()
    mock_logger.level = logging.INFO
    config_manager.logger = mock_logger

    processor = DummyBatchProcessor(
        hook_manager=mock_hook_manager,
        config_manager=config_manager,
        logger=mock_logger,
    )
    processor.execute()

    # execute_hooks が6回呼ばれたか？
    assert mock_hook_manager.execute_hooks.call_count == 6

    actual_calls = [args[0].value for args, kwargs in mock_hook_manager.execute_hooks.call_args_list]
    expected_calls = [
        "pre_setup",
        "post_setup",
        "pre_process",
        "post_process",
        "pre_cleanup",
        "post_cleanup",
    ]
    assert actual_calls == expected_calls

    # ログの確認
    mock_logger.info.assert_any_call("Batch process started.")
    mock_logger.info.assert_any_call("Batch process completed in 0.00 seconds.")

def test_signal_handler_triggers_cleanup():
    # モック構成
    mock_hook_manager = MagicMock()
    mock_config_manager = MagicMock()
    mock_logger = MagicMock()
    mock_config_manager.get_logger.return_value = mock_logger
    mock_config_manager.config = {}

    # processor と handler のセットアップ
    processor = DummyBatchProcessor(
        hook_manager=mock_hook_manager,
        config_manager=mock_config_manager,
        logger=mock_logger,
    )
    signal_handler = SignalHandler(shutdown_callback=processor.cleanup, logger=mock_logger)

    # 疑似的に SIGINT を送る（本物の OS シグナルではない）
    signal_handler._handle_shutdown(signal.SIGINT, None)

    # cleanup() が呼ばれたか
    assert processor.cleanup_called is True

    # ログが出ているか
    expected_msg = f"Received shutdown signal {signal.Signals(signal.SIGINT).name}. Executing cleanup..."
    mock_logger.info.assert_any_call(expected_msg)
