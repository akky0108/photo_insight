import signal
from typing import Callable


class SignalHandler:
    def __init__(self, shutdown_callback: Callable[[], None], logger=None):
        self.shutdown_callback = shutdown_callback
        self.logger = logger
        self._is_registered = False

    def register(self):
        if self._is_registered:
            return

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        self._is_registered = True

        if self.logger:
            self.logger.info("Signal handlers registered for SIGINT and SIGTERM.")

    def _handle_shutdown(self, signum, frame):
        signal_name = (
            signal.Signals(signum).name
            if signum in signal.Signals.__members__.values()
            else str(signum)
        )
        if self.logger:
            self.logger.info(
                f"Received shutdown signal {signal_name}. Executing cleanup..."
            )
        self.shutdown_callback()

    # テスト用にアクセスできるようにプロパティを追加（任意）
    @property
    def handle_shutdown(self):
        return self._handle_shutdown
