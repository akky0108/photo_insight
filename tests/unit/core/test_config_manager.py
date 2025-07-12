import os
import time
import tempfile
import yaml
import logging
import pytest
from unittest.mock import MagicMock
from batch_framework.core.config_manager import ConfigManager


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
    config = {"batch": {"memory_threshold": config_value}} if config_value is not None else {}

    logger_name = "test_logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    manager = ConfigManager(config_path=None)
    manager.config = config
    manager.logger = logger  # 正しい logger を設定

    with caplog.at_level(logging.WARNING, logger=logger_name):
        result = manager.get_memory_threshold(default=90)

    assert result == expected
    if log_message:
        assert any(log_message in message for message in caplog.messages), caplog.messages


def test_load_config_from_file():
    sample_config = {"batch_size": 50}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.dump(sample_config, tmp)
        tmp_path = tmp.name

    cm = ConfigManager(config_path=tmp_path)
    assert cm.config["batch_size"] == 50

    os.remove(tmp_path)


def test_reload_config():
    sample_config1 = {"batch_size": 10}
    sample_config2 = {"batch_size": 99}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.dump(sample_config1, tmp)
        tmp_path = tmp.name

    cm = ConfigManager(config_path=tmp_path)
    assert cm.config["batch_size"] == 10

    # 設定ファイルを変更
    with open(tmp_path, "w") as f:
        yaml.dump(sample_config2, f)

    cm.reload_config()
    assert cm.config["batch_size"] == 99

    os.remove(tmp_path)


def test_logger_called_on_load():
    sample_config = {"batch_size": 15}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.dump(sample_config, tmp)
        tmp_path = tmp.name

    mock_logger = MagicMock()

    manager = ConfigManager(config_path=tmp_path, logger=mock_logger)

    # logger.infoが期待通りの引数で呼ばれているかチェック
    mock_logger.info.assert_any_call(f"Loading configuration from {tmp_path}")

    os.remove(tmp_path)


def test_config_change_triggers_callback():
    triggered = False

    def mock_callback():
        nonlocal triggered
        triggered = True

    sample_config = {"batch_size": 20}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.dump(sample_config, tmp)
        tmp_path = tmp.name

    cm = ConfigManager(config_path=tmp_path)
    cm.start_watching(mock_callback)

    # simulate file change
    time.sleep(1.1)
    with open(tmp_path, "w") as f:
        yaml.dump({"batch_size": 30}, f)

    time.sleep(1.5)

    cm.stop_watching()
    assert triggered is True

    os.remove(tmp_path)


def test_stop_watching_does_not_raise():
    sample_config = {"batch_size": 42}
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".yaml") as tmp:
        yaml.dump(sample_config, tmp)
        tmp_path = tmp.name

    cm = ConfigManager(config_path=tmp_path)
    cm.start_watching(lambda: None)
    cm.stop_watching()

    os.remove(tmp_path)


