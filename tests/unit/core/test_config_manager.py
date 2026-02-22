import os
import tempfile
import yaml
import logging
import pytest
from unittest.mock import MagicMock

from photo_insight.batch_framework.core.config_manager import (
    ConfigManager,
    NullWatchFactory,
    DefaultConfigResolver,
)


@pytest.mark.parametrize(
    "config_value,expected,log_message",
    [
        (95, 95, None),
        ("85", 85, None),
        (0, 90, "Invalid memory_threshold: 0"),
        (101, 90, "Invalid memory_threshold: 101"),
        ("invalid", 90, "Invalid memory_threshold format: invalid"),
        (None, 90, None),
    ],
)
def test_get_memory_threshold_behavior(config_value, expected, log_message, caplog):
    # ここは「ファイル読み込み不要」なので auto_load=False でOK
    config = {"batch": {"memory_threshold": config_value}} if config_value is not None else {}

    logger_name = "test_logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    manager = ConfigManager(
        config_path="dummy.yaml",  # resolve を通すためだけ（auto_load=False なら読まない）
        logger=logger,
        watch_factory=NullWatchFactory(),
        resolver=DefaultConfigResolver(strict_missing=False),  # dummy.yaml が無くても落ちない
        auto_load=False,
    )
    manager.config = config

    with caplog.at_level(logging.WARNING, logger=logger_name):
        result = manager.get_memory_threshold(default=90)

    assert result == expected
    if log_message:
        assert any(log_message in message for message in caplog.messages), caplog.messages


def test_load_config_from_file():
    sample_config = {"batch_size": 50}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.safe_dump(sample_config, tmp)
        tmp_path = tmp.name

    cm = ConfigManager(
        config_path=tmp_path,
        watch_factory=NullWatchFactory(),  # watch 無効で安定
    )
    assert cm.config["batch_size"] == 50

    os.remove(tmp_path)


def test_reload_config():
    sample_config1 = {"batch_size": 10}
    sample_config2 = {"batch_size": 99}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.safe_dump(sample_config1, tmp)
        tmp_path = tmp.name

    cm = ConfigManager(
        config_path=tmp_path,
        watch_factory=NullWatchFactory(),
    )
    assert cm.config["batch_size"] == 10

    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_config2, f)

    cm.reload_config()
    assert cm.config["batch_size"] == 99

    os.remove(tmp_path)


def test_logger_called_on_load():
    sample_config = {"batch_size": 15}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.safe_dump(sample_config, tmp)
        tmp_path = tmp.name

    mock_logger = MagicMock()

    _ = ConfigManager(
        config_path=tmp_path,
        logger=mock_logger,
        watch_factory=NullWatchFactory(),
    )

    # 新実装は "Loading configuration from: <paths...>" 形式
    # assert_any_call で完全一致より、部分一致にして将来の拡張にも強くする
    called = False
    for c in mock_logger.info.call_args_list:
        msg = str(c.args[0]) if c.args else ""
        if "Loading configuration from:" in msg and tmp_path in msg:
            called = True
            break
    assert called, mock_logger.info.call_args_list

    os.remove(tmp_path)


@pytest.mark.skip(reason="watchdog に依存しCIでflakyになりやすい。必要なら統合テストへ。")
def test_config_change_triggers_callback():
    # どうしてもやるなら WatchdogFactory を明示し、
    # CIではスキップ、ローカルでのみ実行推奨。
    ...


def test_stop_watching_does_not_raise():
    sample_config = {"batch_size": 42}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.safe_dump(sample_config, tmp)
        tmp_path = tmp.name

    cm = ConfigManager(
        config_path=tmp_path,
        watch_factory=NullWatchFactory(),
    )
    cm.start_watching(lambda: None)  # NullWatch なので副作用なし
    cm.stop_watching()

    os.remove(tmp_path)
