# tests/integration/dummy_batch_processor.py

from batch_framework.base_batch import BaseBatchProcessor


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
        self.batch_size = 2

    def get_data(self):
        # 6アイテム → 3バッチ（2件ずつ）
        return [{"id": i} for i in range(6)]

    def _process_batch(self, batch):
        ids = [item["id"] for item in batch]
        self.logger.debug(f"Processing batch with IDs: {ids}")
        start_id = batch[0]["id"]
        if start_id == 2:
            self.logger.debug("Simulating batch failure.")
            raise ValueError("Simulated batch failure")
        self.logger.debug("Batch succeeded.")
        self.processed_batches.extend(ids)
