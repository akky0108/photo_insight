import os
import yaml
import json
import signal
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Optional, Callable, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from log_util import Logger
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

    def on_modified(self, event):
        if event.src_path == self.processor.config_path:
            self.processor.logger.info(f"Config file {event.src_path} has been modified. Reloading...")
            self.processor.reload_config()

class BaseBatchProcessor(ABC):
    """バッチ処理の基底クラス"""

    def __init__(self, config_path: Optional[str] = None, logger: Optional[Logger] = None, max_workers: int = 2, max_process_count: Optional[int] = None):
        load_dotenv()
        self.project_root = os.getenv('PROJECT_ROOT') or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.default_config = {
            "config_path": os.path.join(self.project_root, "config", "config.yaml"),
            "setting_1": "default_value_1",
            "setting_2": "default_value_2"
        }

        self.logger = logger if logger else Logger(project_root=self.project_root, logger_name=self.__class__.__name__).get_logger()
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

    def load_config(self, config_path: str) -> None:
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}. Using default settings.")
            
            if config_path.endswith(('.yaml', '.yml')):
                with open(config_path, 'r') as config_file:
                    loaded_config = yaml.safe_load(config_file)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as config_file:
                    loaded_config = json.load(config_file)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")

            self.config.update(loaded_config)
            self.logger.info(f"Configuration loaded from {config_path}.")
        
        except Exception as e:
            self.handle_error(f"Error loading configuration from '{config_path}': {e}")

    def reload_config(self, config_path: Optional[str] = None) -> None:
        self.logger.info("Reloading configuration.")
        if config_path:
            self.config_path = config_path
        self.load_config(self.config_path)

    def _start_config_watcher(self, config_path: str) -> None:
        event_handler = ConfigChangeHandler(self)
        observer = Observer()
        observer.schedule(event_handler, path=os.path.dirname(config_path), recursive=False)
        observer.start()
        self.logger.info(f"Started watching for changes in {config_path}.")

    def add_hook(self, hook_type: HookType, func: Callable[[], None], priority: int = 0) -> None:
        hook_list = self.hooks[hook_type]
        for index, (existing_priority, _) in enumerate(hook_list):
            if existing_priority < priority:
                hook_list.insert(index, (priority, func))
                return
        hook_list.append((priority, func))

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
        """各フェーズの前後フックとフェーズ自体の実行を管理"""
        phase_name = pre_hook_type.name.split('_')[1].lower()
        self.logger.info(f"Starting {phase_name} phase.")
        phase_start_time = time.time()

        try:
            self._execute_hooks(pre_hook_type)
            phase_function()
            self._execute_hooks(post_hook_type)
        except Exception as e:
            self.logger.error(f"{phase_name.capitalize()} phase encountered an error: {e}")
            errors.append(e)
        finally:
            phase_duration = time.time() - phase_start_time
            self.logger.info(f"{phase_name.capitalize()} phase completed in {phase_duration:.2f} seconds.")

    @abstractmethod
    def setup(self) -> None:
        """セットアップフェーズ。サブクラスで実装"""
        pass

    @abstractmethod
    def process(self) -> None:
        """個別のタスクを実行。サブクラスで実装"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """クリーンアップフェーズ。サブクラスで実装"""
        pass
