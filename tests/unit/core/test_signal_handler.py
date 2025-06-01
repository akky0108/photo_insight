import signal
import builtins
from unittest.mock import MagicMock
import pytest
import batch_framework.core.signal_handler as signal_handler_module
from batch_framework.core.signal_handler import SignalHandler


def test_signal_handler_registration(monkeypatch):
    registered = {}

    def dummy_signal(sig, func):
        registered[sig] = func

    monkeypatch.setattr(signal_handler_module.signal, "signal", dummy_signal)

    mock_callback = MagicMock()
    logger = MagicMock()
    handler = SignalHandler(mock_callback, logger=logger)

    handler.register()

    assert signal.SIGINT in registered
    assert signal.SIGTERM in registered
    logger.info.assert_called_with("Signal handlers registered for SIGINT and SIGTERM.")


def test_signal_handler_invokes_cleanup_and_exit(monkeypatch):
    monkeypatch.setattr(signal_handler_module.signal, "signal", lambda sig, func: None)

    mock_callback = MagicMock()
    logger = MagicMock()
    handler = SignalHandler(mock_callback, logger=logger)

    handler._handle_shutdown(signal.SIGINT, None)

    logger.info.assert_called_with("Received shutdown signal SIGINT. Executing cleanup...")
    mock_callback.assert_called_once()
