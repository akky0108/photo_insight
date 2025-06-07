import os
import time
import tempfile
import yaml
from unittest.mock import MagicMock
from batch_framework.core.config_manager import ConfigManager


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
