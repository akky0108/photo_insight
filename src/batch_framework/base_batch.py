import os
import yaml
import signal
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Optional, Callable, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class HookType(Enum):
    """フックの種類を定義する列挙型"""
    PRE_SETUP = 'pre_setup'
    POST_SETUP = 'post_setup'
    PRE_PROCESS = 'pre_process'
    POST_PROCESS = 'post_process'
    PRE_CLEANUP = 'pre_cleanup'
    POST_CLEANUP = 'post_cleanup'


class ConfigChangeHandler(FileSystemEventHandler):
    """コンフィグファイルの変更を監視するハンドラー"""
    def __init__(self, processor: 'BaseBatchProcessor'):
        self.processor = processor
        self._last_modified_time = 0  # デバウンス用

    def on_modified(self, event):
        now = time.time()
        if now - self._last_modified_time < 1.0:  # 1秒以内の再発火を無視
            return
        self._last_modified_time = now
        if event.src_path == self.processor.config_path:
            self.processor.logger.info(f"Config file {event.src_path} has been modified. Reloading...")
            self.processor.reload_config()


class BaseBatchProcessor(ABC):
    """バッチ処理の基底クラス"""

    def __init__(self, config_path: Optional[str] = None, max_workers: int = 2, max_process_count: Optional[int] = None):
        load_dotenv()
        self.project_root = os.getenv('PROJECT_ROOT') or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.default_config = {
            "config_path": os.path.join(self.project_root, "config", "config.yaml"),
            "batch_size": 100
        }

        # ロギングの初期化（_get_default_loggerを使って統一）
        self.logger = self._get_default_logger()

        self.max_workers = max_workers
        self.config_path = config_path or self.default_config["config_path"]
        self.config = self.default_config.copy()
        self.max_process_count = max_process_count
        self.processed_count = 0

        # フックの初期化
        self.hooks: Dict[HookType, List[Tuple[int, Callable[[], None]]]] = {hook_type: [] for hook_type in HookType}

        # コンフィグの読み込みとウォッチャーの開始
        self.load_config(self.config_path)
        self._start_config_watcher(self.config_path)

        # シグナルハンドリングの設定
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _get_default_logger(self):
        from utils.app_logger import Logger  # 遅延インポート（循環依存回避用）
        return Logger(
            project_root=self.project_root,
            logger_name=self.__class__.__name__
        ).get_logger()

    def load_config(self, config_path: str) -> None:
        """設定ファイルの読み込み処理"""
        try:
            self.logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                self.config.clear()  # 追加: 古い設定をクリア
                self.config.update(yaml.safe_load(f))
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def reload_config(self, config_path: Optional[str] = None) -> None:
        self.logger.info("Reloading configuration.")
        if config_path:
            self.config_path = config_path
        self.load_config(self.config_path)

    def _start_config_watcher(self, config_path: str) -> None:
        """設定ファイル変更監視を開始"""
        try:
            self.observer = Observer()
            event_handler = ConfigChangeHandler(self)
            self.observer.schedule(event_handler, path=os.path.dirname(config_path), recursive=False)
            self.observer.start()
            self.logger.info(f"Watching configuration changes in {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to start config watcher: {e}")
            raise

    def add_hook(self, hook_type: HookType, func: Callable[..., None], priority: int = 0) -> None:
        hook_list = self.hooks[hook_type]
        hook_list.append((priority, func))
        hook_list.sort(reverse=True, key=lambda x: x[0])

    def _execute_hooks(self, hook_type: HookType) -> None:
        """指定されたフックタイプのフックを実行"""
        hooks = self.hooks[hook_type]
        errors = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(hook[1]) for hook in hooks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error during hook execution: {e}")
                    errors.append(e)

        if errors:
            self.handle_error(f"Errors encountered during {hook_type.value} hook execution: {errors}")

    def handle_error(self, message: str, raise_exception: bool = False) -> None:
        self.logger.error(message)
        if raise_exception:
            raise RuntimeError(message)

    def _handle_shutdown(self, signum, frame):
        self.logger.info(f"Received shutdown signal {signum}.")
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
        self.cleanup()

    def execute(self) -> None:
        """バッチ処理のメインエントリポイント"""
        self.start_time = time.time()
        self.logger.info("Batch process started.")
        errors = []

        try:
            # PRE_SETUP, POST_SETUP フェーズ
            self._execute_phase(HookType.PRE_SETUP, HookType.POST_SETUP, self.setup, errors)

            # PRE_PROCESS, POST_PROCESS フェーズ
            self._execute_phase(HookType.PRE_PROCESS, HookType.POST_PROCESS, self.process, errors)

        finally:
            # PRE_CLEANUP, POST_CLEANUP フェーズ
            self._execute_phase(HookType.PRE_CLEANUP, HookType.POST_CLEANUP, self.cleanup, errors)
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.logger.info(f"Batch process completed in {duration:.2f} seconds.")

            if errors:
                self.handle_error(f"Batch process encountered errors: {errors}", raise_exception=True)

    def _execute_phase(self, pre_hook_type: HookType, post_hook_type: HookType, phase_function: Callable[[], None], errors: List[Exception]) -> None:
        phase_name = pre_hook_type.name.split('_')[1].lower()
        self._run_phase_hooks(pre_hook_type, errors)
        self._run_phase_function(phase_function, phase_name, errors)
        self._run_phase_hooks(post_hook_type, errors)

    def _run_phase_hooks(self, hook_type: HookType, errors: List[Exception]) -> None:
        try:
            self._execute_hooks(hook_type)
        except Exception as e:
            self.logger.error(f"Error in {hook_type.name} hooks: {e}")
            errors.append(e)

    def _run_phase_function(self, func: Callable[[], None], name: str, errors: List[Exception]) -> None:
        try:
            start_time = time.time()
            self.logger.info(f"Executing {name} phase.")
            func()
            duration = time.time() - start_time
            self.logger.info(f"{name.capitalize()} phase completed in {duration:.2f} seconds.")
        except Exception as e:
            self.logger.error(f"Error during {name} phase: {e}")
            errors.append(e)

    def setup(self) -> None:
        """共通のセットアップ処理"""
        self.logger.info("Executing common setup tasks in BaseBatchProcessor.")

    def process(self) -> None:
        """共通のプロセス処理フロー"""
        self.logger.info("Executing common batch processing tasks in BaseBatchProcessor.")

        batches = self._generate_batches()
        for i, batch in enumerate(batches):
            self.logger.info(f"Processing batch {i + 1}...")
            self._process_batch(batch)

    @abstractmethod
    def _process_batch(self, batch: List[Dict]) -> None:
        """バッチ単位の具体的処理を子クラスで実装"""
        pass

    def _generate_batches(self) -> List[List[Dict]]:
        """データを分割してバッチを生成"""
        batch_size = self.config.get("batch_size", 100)
        return [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]

    def cleanup(self) -> None:
        """共通のクリーンアップ処理"""
        self.logger.info("Executing common cleanup tasks in BaseBatchProcessor.")
