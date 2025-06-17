# tests/integration/dummy_batch_processor.py

from batch_framework.base_batch import BaseBatchProcessor
from batch_framework.core.hook_manager import HookManager


class DummyBatchProcessor(BaseBatchProcessor):
    def __init__(self, hook_manager, config_manager, signal_handler=None, logger=None):
        super().__init__(
            hook_manager=hook_manager,
            config_manager=config_manager,
            signal_handler=signal_handler,
        )
        if logger:
            self.logger = logger
        self.setup_called = False
        self.process_called = False
        self.cleanup_called = False

    def setup(self):
        self.setup_called = True

    def process(self):
        self.process_called = True

    def _process_batch(self, batch):
        # ダミー実装（何もしない）
        pass

    def get_data(self):
        # ダミー実装：空のデータを返す
        return []

    def cleanup(self):
        self.cleanup_called = True

class DummyBatchProcessorWithFailingBatch(BaseBatchProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_batches = []

    def get_data(self):
        # 6アイテム → 3バッチ（2件ずつ）
        return [{"id": i} for i in range(6)]

    def _process_batch(self, batch):
        self.logger.debug(f"DEBUG: _process_batch called with batch: {[item['id'] for item in batch]}")
        start_id = batch[0]["id"]
        if start_id == 2:  # 2番目のバッチ（id:2,3）は失敗させる
            raise ValueError("Simulated batch failure")
        self.processed_batches.extend([item["id"] for item in batch])
