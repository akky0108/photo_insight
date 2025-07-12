import os
import yaml
import time
from typing import Callable, Optional
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[], None]):
        self.callback = callback
        self._last_modified_time = 0

    def on_modified(self, event):
        now = time.time()
        if now - self._last_modified_time < 1.0:
            return
        self._last_modified_time = now
        self.callback()


class ConfigManager:
    """
    設定ファイルの読み込み、変更監視、再読み込みを管理するクラス。
    """

    def __init__(self, config_path: Optional[str] = None, logger=None):
        """
        Args:
            config_path (Optional[str]): 設定ファイルのパス。
            logger (Optional[Logger]): ロガーインスタンス。
        """
        load_dotenv()
        self.logger = logger
        self.project_root = os.getenv("PROJECT_ROOT") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.config_path = config_path or os.path.join(
            self.project_root, "config", "config.yaml"
        )
        self.config = {}
        self.observer = None

        self.load_config(self.config_path)

    def load_config(self, config_path: str) -> None:
        """
        設定ファイルを読み込む。
        """
        try:
            if self.logger:
                self.logger.info(f"Loading configuration from {config_path}")
            with open(config_path, "r") as f:
                self.config.clear()
                data = yaml.safe_load(f)
                if data:
                    self.config.update(data)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load config: {e}")
            raise

    def reload_config(self) -> None:
        """
        設定ファイルを再読み込みする。
        """
        if self.logger:
            self.logger.info("Reloading configuration.")
        self.load_config(self.config_path)

    def start_watching(self, on_change_callback: Callable[[], None]) -> None:
        """
        設定ファイルの変更を監視する。

        Args:
            on_change_callback (Callable[[], None]): 変更時に呼び出されるコールバック関数。
        """
        try:
            handler = ConfigChangeHandler(callback=on_change_callback)
            self.observer = Observer()
            self.observer.schedule(
                handler, path=os.path.dirname(self.config_path), recursive=False
            )
            self.observer.start()
            if self.logger:
                self.logger.info(
                    f"Watching configuration changes in {self.config_path}"
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to start config watcher: {e}")
            raise

    def stop_watching(self) -> None:
        """
        設定ファイルの変更監視を停止する。
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()

    def get_config(self) -> dict:
        """
        設定値を取得する。

        Returns:
            dict: 現在の設定。
        """
        return self.config

    def get_logger(self, logger_name: Optional[str] = None):
        """
        デフォルトのロガーを生成・返却する。
        外部から logger を渡していない場合に使用される。

        Args:
            logger_name (Optional[str]): 任意のロガー名（クラス名など）

        Returns:
            logging.Logger: 初期化されたロガー
        """
        if not logger_name:
            logger_name = self.__class__.__name__

        from utils.app_logger import Logger

        return Logger(
            project_root=self.project_root, logger_name=logger_name
        ).get_logger()

    def get_memory_threshold(self, default: int = 90) -> int:
        """
        メモリ使用率のしきい値（%）を設定ファイルから取得する。
        値が 1～100 の範囲外であれば default を返す。

        Args:
            default (int): デフォルト値（通常は90）

        Returns:
            int: 使用するしきい値（1～100）
        """
        value = self.config.get("batch", {}).get("memory_threshold", default)
        try:
            value = int(value)
            if 1 <= value <= 100:
                return value
            else:
                if self.logger:
                    self.logger.warning(f"Invalid memory_threshold: {value}. Using default: {default}")
        except (ValueError, TypeError):
            if self.logger:
                self.logger.warning(f"Invalid memory_threshold format: {value}. Using default: {default}")
        return default