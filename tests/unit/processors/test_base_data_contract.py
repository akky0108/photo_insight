import yaml
from typing import Any, Dict, List
from unittest.mock import MagicMock

from batch_framework.base_batch import BaseBatchProcessor


def _write_min_config(tmp_path) -> str:
    cfg = {"batch": {"memory_threshold": 90}, "debug": {"log_summary_detail": False}}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return str(p)


class DataContractProcessor(BaseBatchProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_calls = 0

    def load_data(self) -> List[Dict[str, Any]]:
        self.load_calls += 1
        return [{"x": 1}, {"x": 2}, {"x": 3}]

    def _process_batch(self, batch):
        return [{"status": "success", "score": 1.0, "row": b} for b in batch]


def test_setup_calls_load_data_once(tmp_path):
    config_path = _write_min_config(tmp_path)
    p = DataContractProcessor(config_path=config_path, logger=MagicMock(), max_workers=2)

    p.setup()
    assert p.load_calls == 1
    assert isinstance(p.data, list)
    assert len(p.data) == 3


def test_process_does_not_reload_data(tmp_path):
    config_path = _write_min_config(tmp_path)
    p = DataContractProcessor(config_path=config_path, logger=MagicMock(), max_workers=2)

    p.setup()
    assert p.load_calls == 1

    p.process()
    assert p.load_calls == 1
