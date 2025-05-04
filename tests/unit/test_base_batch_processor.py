import os
import sys
import pytest

print(sys.path)

from unittest.mock import patch, mock_open, MagicMock
from batch_framework.base_batch import BaseBatchProcessor
from batch_framework.base_batch import HookType

class DummyBatchProcessor(BaseBatchProcessor):
    def __init__(self, *args, **kwargs):
        # 副作用のない初期化
        self.project_root = os.getcwd()
        self.default_config = {"batch_size": 100}
        self.config_path = kwargs.get("config_path", "dummy.yaml")
        self.config = self.default_config.copy()
        self.logger = MagicMock()
        self.hooks = {hook_type: [] for hook_type in HookType}
        self.max_workers = 2
        self.max_process_count = None
        self.processed_count = 0
    def _process_batch(self, batch): pass

@patch("builtins.open", new_callable=mock_open, read_data="batch_size: 50")
@patch("yaml.safe_load", return_value={"batch_size": 50})
def test_load_config_success(mock_yaml, mock_file):
    processor = DummyBatchProcessor()
    processor.load_config("test_config.yaml")

    print("open() call list:", mock_file.call_args_list)
    mock_file.assert_any_call("test_config.yaml", 'r')
    mock_yaml.assert_called_once()
    assert processor.config["batch_size"] == 50

@patch("builtins.open", new_callable=mock_open)
@patch("yaml.safe_load", side_effect=Exception("File read error"))
def test_load_config_failure(mock_yaml, mock_file):
    processor = DummyBatchProcessor()
    with pytest.raises(Exception, match="File read error"):
        processor.load_config("test_config.yaml")

def test_add_hook_priority():
    from batch_framework.base_batch import HookType

    class DummyBatchProcessor(BaseBatchProcessor):
        def _process_batch(self, batch):
            pass

    processor = DummyBatchProcessor(config_path=None)

    execution_order = []

    def hook1(): execution_order.append("low")
    def hook2(): execution_order.append("high")

    processor.add_hook(HookType.PRE_SETUP, hook1, priority=1)
    processor.add_hook(HookType.PRE_SETUP, hook2, priority=10)

    processor._execute_hooks(HookType.PRE_SETUP)

    assert execution_order == ["high", "low"]


def test_execute_hooks_runs_all():
    from batch_framework.base_batch import HookType

    class DummyBatchProcessor(BaseBatchProcessor):
        def _process_batch(self, batch):
            pass

    processor = DummyBatchProcessor(config_path=None)

    called_flags = []

    def hook1(): called_flags.append("hook1")
    def hook2(): called_flags.append("hook2")

    processor.add_hook(HookType.PRE_PROCESS, hook1)
    processor.add_hook(HookType.PRE_PROCESS, hook2)

    processor._execute_hooks(HookType.PRE_PROCESS)

    assert "hook1" in called_flags
    assert "hook2" in called_flags


@patch('builtins.open', new_callable=mock_open, read_data='batch_size: 200')
@patch('yaml.safe_load', return_value={"batch_size": 200})
@patch.object(BaseBatchProcessor, '_start_config_watcher')  # 追加ここ！
def test_reload_config_with_new_path(mock_watcher, mock_yaml, mock_file):
    class DummyBatchProcessor(BaseBatchProcessor):
        def _process_batch(self, batch):
            pass

    processor = DummyBatchProcessor(config_path="original.yaml")
    processor.reload_config("new_path.yaml")

    assert processor.config_path == "new_path.yaml"
    assert processor.config["batch_size"] == 200

