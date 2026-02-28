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


# =============================
# evaluation_rank fixtures
# =============================
import json
from typing import Any, Dict, List


@pytest.fixture()
def required_columns() -> List[str]:
    # INPUT_REQUIRED_COLUMNS を SSOT として利用
    from photo_insight.batch_processor.evaluation_rank.contract import INPUT_REQUIRED_COLUMNS

    return list(INPUT_REQUIRED_COLUMNS)


def _default_value_for_eval(col: str, i: int) -> Any:
    # 最低限の型/表現だけ合わせる（normalize層が吸える形）
    if col in ("file_name", "group_id", "subgroup_id", "shot_type", "accepted_reason"):
        return f"{col}_{i}"

    if col == "faces":
        # _safe_parse_faces が json として読める形式
        return json.dumps(
            [
                {
                    "confidence": 0.9,
                    "eye_closed_prob": 0.1,
                    "eye_lap_var": 0.2,
                    "eye_patch_size": 32,
                }
            ]
        )

    if col in ("face_detected", "full_body_detected"):
        return "true"

    if col == "accepted_flag":
        return "0"

    # それ以外は数値っぽく
    return 0.0


@pytest.fixture()
def minimal_required_row(required_columns: List[str]) -> Dict[str, Any]:
    row = {c: _default_value_for_eval(c, 0) for c in required_columns}
    # よく使うキーは明示
    row["file_name"] = "IMG_0001.NEF"
    row["group_id"] = "g1"
    row["subgroup_id"] = "sg1"
    row["shot_type"] = "portrait"
    row["accepted_flag"] = "0"
    row["accepted_reason"] = ""
    return row


@pytest.fixture()
def minimal_required_rows(minimal_required_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    r1 = dict(minimal_required_row)
    r2 = dict(minimal_required_row)
    r2["file_name"] = "IMG_0002.NEF"
    r2["subgroup_id"] = "sg2"
    return [r1, r2]
