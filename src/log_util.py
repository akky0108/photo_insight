# Log_Util.py

import logging
import logging.config
import os
import yaml
import threading
import atexit
from typing import Optional

class AppLogger:
    _instance = None  # シングルトンインスタンス
    _lock = threading.Lock()  # スレッドセーフのためのロック

    def __new__(cls, *args, **kwargs):
        """ シングルトンインスタンスを作成、または既存インスタンスを返す """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, project_root: Optional[str] = None, config_file: Optional[str] = None, logger_name: str = 'MyAppLogger'):
        if getattr(self, '_initialized', False):
            return  # 既に初期化済みの場合は何もしない

        # プロジェクトルートからデフォルトのconfig_fileパスを指定
        if config_file is None:
            if project_root:
                config_file = os.path.join(project_root, 'config', 'logging_config.yaml')
            else:
                config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'logging_config.yaml')

        # YAMLファイルからロギング設定を読み込む
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if isinstance(config, dict):
                        logging.config.dictConfig(config)
                    else:
                        raise ValueError("Invalid config format")
                print(f"Logging configured from {config_file}.")
            except (yaml.YAMLError, ValueError, FileNotFoundError) as e:
                print(f"Error loading logging config: {e}. Using default logging settings.")
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
