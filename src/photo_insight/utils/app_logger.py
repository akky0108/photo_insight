"""
utils.app_logger

プロジェクト全体で共通して使えるログユーティリティ。

- ロギング設定は YAML ファイルで読み込む
- Singleton パターンで logger インスタンスを管理
- メトリクス出力や cleanup 処理など便利機能付き

例:
    logger = AppLogger(project_root=".", logger_name="MyLogger").get_logger()
    logger.info("Hello!")
"""

import logging
import logging.config
import os
import yaml
import threading
import atexit
from typing import Optional


class AppLogger:
    """
    アプリケーション用のロガー管理クラス（Singleton）。

    ログ設定は YAML ファイルから読み込み、指定された logger_name に基づいてロガーを生成。
    cleanup により、プログラム終了時にハンドラのクローズも行う。

    引数:
        project_root (Optional[str]): プロジェクトルートパス（config/logging_config.yaml を探すのに使用）
        config_file (Optional[str]): 明示的にロギング設定ファイルを指定したい場合のパス
        logger_name (str): 作成するロガーの名前（クラス名などを推奨）

    使い方:
        logger = AppLogger(project_root=".", logger_name="MyLogger").get_logger()
        logger.info("起動完了")
    """

    _instance = None  # シングルトンインスタンス
    _lock = threading.Lock()  # スレッドセーフのためのロック

    def __new__(cls, *args, **kwargs):
        """シングルトンインスタンスを作成、または既存インスタンスを返す"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        project_root: Optional[str] = None,
        config_file: Optional[str] = None,
        logger_name: str = "MyAppLogger",
    ):
        if getattr(self, "_initialized", False):
            return  # 既に初期化済みの場合は何もしない

        if project_root is None:
            # カレントディレクトリをプロジェクトルートに設定
            project_root = os.getcwd()

        # プロジェクトルートからデフォルトのconfig_fileパスを指定
        if config_file is None:
            if project_root:
                config_file = os.path.join(project_root, "config", "logging_config.yaml")
            else:
                config_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "config",
                    "logging_config.yaml",
                )

        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # YAMLファイルからロギング設定を読み込む
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                    if isinstance(config, dict):
                        logging.config.dictConfig(config)
                    else:
                        raise ValueError("Invalid config format")
                print(f"Logging configured from {config_file}.")
            except (yaml.YAMLError, ValueError, FileNotFoundError) as e:
                print(f"Error loading logging config: {e}. " f"Using default logging settings.")
                logging.basicConfig(level=logging.DEBUG)
        else:
            print(f"Config file {config_file} not found. Using default logging settings.")
            logging.basicConfig(level=logging.DEBUG)

        # ロガーを設定
        self.logger = logging.getLogger(logger_name)

        # プログラム終了時にクリーンアップを登録
        atexit.register(self.cleanup)

        # 初期化フラグを立てる
        self._initialized = True

    def _log(self, level: int, message: str):
        """ログメッセージを出力"""
        if self.isEnabledFor(level):
            self.logger.log(level, message)

    def info(self, message: str):
        """INFOレベルのログメッセージを出力"""
        self._log(logging.INFO, message)

    def error(self, message: str):
        """ERRORレベルのログメッセージを出力"""
        self._log(logging.ERROR, message)

    def debug(self, message: str):
        """DEBUGレベルのログメッセージを出力"""
        self._log(logging.DEBUG, message)

    def warning(self, message: str):
        """WARNINGレベルのログメッセージを出力"""
        self._log(logging.WARNING, message)

    def critical(self, message: str):
        """CRITICALレベルのログメッセージを出力"""
        self._log(logging.CRITICAL, message)

    def log_metric(self, metric_name: str, score: float) -> None:
        """メトリック名とスコアをINFOレベルでログに出力"""
        self.info(f"{metric_name.capitalize()} score: {score}")

    def change_logger_name(self, new_name: str):
        """ロガーネームを変更"""
        # 現在のロガーを削除
        logging.Logger.manager.loggerDict.pop(self.logger.name, None)
        self.logger = logging.getLogger(new_name)

        # 新しいロガーにハンドラを再設定
        # 必要なら、ここでハンドラの再設定処理を追加

    def get_logger(self):
        """現在のロガーを返す"""
        return self.logger

    def cleanup(self):
        """プログラム終了時にハンドラをクリーンアップ"""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

    def isEnabledFor(self, level: int) -> bool:
        """特定のログレベルが有効かどうかを確認"""
        return self.logger.isEnabledFor(level)


# エイリアスとして公開
Logger = AppLogger
