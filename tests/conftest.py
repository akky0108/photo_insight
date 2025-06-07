import sys
import os
import pytest
from unittest.mock import MagicMock

from batch_framework.base_batch import BaseBatchProcessor
from batch_framework.core.hook_manager import HookType

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# === fixtures ===
@pytest.fixture
def fixture_config_path():
    return os.path.join("tests", "fixtures", "test_config.yaml")


@pytest.fixture
def fixture_image_dir():
    return os.path.join("tests", "fixtures", "images")


@pytest.fixture
def fixture_output_dir():
    return os.path.join("tests", "fixtures", "output")


class DummyBatchProcessor(BaseBatchProcessor):
    def _process_batch(self, batch):
        pass

    def __init__(self, *args, **kwargs):
        # 通常のBaseBatchProcessorをモックしやすく簡素化
        self.project_root = os.getcwd()
        self.default_config = {
            "batch_size": 100,
            "output_directory": "tests/output",
        }
        self.config_path = kwargs.get("config_path", "tests/fixtures/test_config.yaml")
        self.config = self.default_config.copy()
        self.logger = MagicMock()
        self.hooks = {hook_type: [] for hook_type in HookType}
        self.max_workers = 2
        self.max_process_count = None
        self.processed_count = 0


@pytest.fixture
def dummy_processor():
    return DummyBatchProcessor()
