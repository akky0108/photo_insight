# tests/utils/test_app_logger.py

import os
import tempfile
import pytest
import logging
from utils.app_logger import AppLogger


def test_singleton_behavior():
    logger1 = AppLogger(project_root=".", logger_name="TestLogger")
    logger2 = AppLogger(project_root=".", logger_name="TestLogger")
    assert logger1 is logger2, "AppLogger should be a singleton"


def test_logger_output_info_level(caplog):
    logger = AppLogger(project_root=".", logger_name="TestLogger").get_logger()
    with caplog.at_level("INFO"):
        logger.info("Test info message")
    assert "Test info message" in caplog.text


def test_logger_output_debug_level(caplog):
    logger = AppLogger(project_root=".", logger_name="TestLogger").get_logger()
    with caplog.at_level("DEBUG"):
        logger.debug("Test debug message")
    assert "Test debug message" in caplog.text


def test_logger_metric_logging(caplog):
    app_logger = AppLogger(project_root=".", logger_name="MetricLogger")
    with caplog.at_level("INFO"):
        app_logger.log_metric("accuracy", 0.95)
    assert "Accuracy score: 0.95" in caplog.text


def test_logger_cleanup():
    app_logger = AppLogger(project_root=".", logger_name="CleanupLogger")
    logger = app_logger.get_logger()
    app_logger.cleanup()
    assert len(logger.handlers) == 0, "All handlers should be removed on cleanup"


def test_log_levels():
    logger = AppLogger(project_root=".", logger_name="TestLogger").get_logger()

    # 各レベルのログが出力されるか確認
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    # 実際にログメッセージが記録されているか（コンソール確認等）
    # 詳細なチェックは出力先に依存します。たとえば、ログ内容をキャプチャして確認する方法も。


def test_log_to_file():
    logger = AppLogger(project_root=".", logger_name="FileLogger").get_logger()

    # ログファイルに出力されているか確認するテスト（仮に 'app.log' として）
    logger.info("This is a file log test")

    # ファイルをチェックして、ログ内容が含まれているか
    log_path = os.path.join("logs", "app.log")
    with open(log_path, "r") as log_file:
        assert "This is a file log test" in log_file.read()


def test_invalid_config_file():
    # 存在しないファイルを指定してエラーが発生するか
    logger = AppLogger(
        project_root=".", config_file="invalid_path.yaml", logger_name="TestLogger"
    )
    logger.info("This should use default config")


def test_change_logger_name():
    app_logger = AppLogger(project_root=".", logger_name="InitialLogger")
    app_logger.logger.info("Initial log")
    app_logger.change_logger_name("NewLogger")
