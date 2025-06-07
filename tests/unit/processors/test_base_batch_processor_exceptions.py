import pytest
from unittest.mock import MagicMock
from batch_framework.base_batch import BaseBatchProcessor, HookType


# ダミー実装
class HookErrorProcessor(BaseBatchProcessor):
    def get_data(self):
        return []

    def _process_batch(self, batch):
        pass


def test_hook_exception_logged(tmp_path):
    processor = HookErrorProcessor()

    # ロガーをモックに差し替え
    processor.logger = MagicMock()

    # 例外を発生させるフック関数
    def failing_hook():
        raise RuntimeError("Hook failure!")

    # フック登録（PRE_SETUP）
    processor.add_hook(HookType.PRE_SETUP, failing_hook)

    # 実行（RuntimeErrorはraiseされない）
    processor.execute()

    # .exception() が呼ばれたことを確認
    assert processor.logger.exception.called

    # ログ内容に "Hook failure" が含まれることを確認（引数文字列に含まれるか）
    found = any(
        "Hook failure" in str(call.args)
        for call in processor.logger.exception.call_args_list
    )
    assert found, "Hook failure に関するログが出力されていません"


class CleanupErrorProcessor(BaseBatchProcessor):
    def get_data(self):
        return []

    def _process_batch(self, batch):
        pass

    def cleanup(self):
        raise RuntimeError("Cleanup failed!")


def test_cleanup_exception_logged(tmp_path):
    processor = CleanupErrorProcessor()
    processor.logger = MagicMock()

    with pytest.raises(RuntimeError):
        processor.execute()

    assert processor.logger.exception.called
    assert any(
        "Cleanup failed" in str(call.args)
        for call in processor.logger.exception.call_args_list
    )
