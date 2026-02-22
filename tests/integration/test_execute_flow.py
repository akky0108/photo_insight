import json
from unittest.mock import MagicMock
from pathlib import Path

from photo_insight.batch_framework.core.config_manager import ConfigManager
from tests.integration.dummy_batch_processor import DummyBatchProcessor

def test_execute_runs_all_phases(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({}))

    config_manager = ConfigManager(config_path=str(config_path))
    processor = DummyBatchProcessor(
        hook_manager=MagicMock(),
        config_manager=config_manager,
    )
    processor.execute()