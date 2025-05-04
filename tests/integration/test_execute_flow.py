import time
from conftest import DummyBatchProcessor
from batch_framework.base_batch import HookType

def test_execute_runs_all_phases():
    processor = DummyBatchProcessor()

    # 各フェーズのマーカー
    phases = []

    processor.setup = lambda: phases.append("setup")
    processor.process = lambda: phases.append("process")
    processor.cleanup = lambda: phases.append("cleanup")

    processor.execute()

    assert phases == ["setup", "process", "cleanup"]
