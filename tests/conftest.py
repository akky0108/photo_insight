import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.core.batch_framework._internal.hook_manager import HookType


# -----------------------------
# repo root (tests/直下固定)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# ★A: Config をテスト用に固定
# -----------------------------
@pytest.fixture(autouse=True)
def _set_test_config_env(monkeypatch):
    # ConfigManager がここを見る
    monkeypatch.setenv("PROJECT_ROOT", str(REPO_ROOT))

    # 既存fixtureで使ってるパスに合わせる（tests/fixtures/test_config.yaml）
    cfg = REPO_ROOT / "tests" / "fixtures" / "test_config.yaml"
    monkeypatch.setenv("CONFIG_PATH", str(cfg))


# -----------------------------
# Heavy / GPU opt-in（将来用）
# -----------------------------
def pytest_addoption(parser):
    parser.addoption("--run-heavy", action="store_true", default=False, help="Run tests marked as heavy")
    parser.addoption("--run-gpu", action="store_true", default=False, help="Run tests marked as gpu")


def pytest_collection_modifyitems(config, items):
    run_heavy = config.getoption("--run-heavy")
    run_gpu = config.getoption("--run-gpu")

    skip_heavy = pytest.mark.skip(reason="need --run-heavy to run")
    skip_gpu = pytest.mark.skip(reason="need --run-gpu to run")

    for item in items:
        if "heavy" in item.keywords and not run_heavy:
            item.add_marker(skip_heavy)
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)


# -----------------------------
# Common fixtures
# -----------------------------
@pytest.fixture
def fixture_config_path():
    return os.path.join("tests", "fixtures", "test_config.yaml")


@pytest.fixture
def fixture_image_dir():
    return os.path.join("tests", "fixtures", "images")


@pytest.fixture
def fixture_output_dir():
    return os.path.join("tests", "fixtures", "output")


# -----------------------------
# Dummy Processor
# -----------------------------
class DummyBatchProcessor(BaseBatchProcessor):
    def __init__(self, *args, **kwargs):
        # BaseBatchProcessor をモックしやすく簡素化
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

    def _process_batch(self, batch):
        pass


@pytest.fixture
def dummy_processor():
    return DummyBatchProcessor()
