import os
import yaml
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Optional, Callable, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from log_util import Logger
from enum import Enum
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import signal

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
        """コンフィグファイルが変更された際に呼び出されるメソッド"""
        if event.src_path == self.processor.config_path:
            self.processor.logger.info(f"Config file {event.src_path} has been modified. Reloading...")
            self.processor.reload_config()

class BaseBatchProcessor(ABC):
    """バッチ処理の基底クラス。フックやフェーズ管理を行い、指定された件数で終了するオプションを追加"""

    default_config = {
        "config_path": "./config.yaml",
        "setting_1": "default_value_1", 
        "setting_2": "default_value_2"
    }

    def __init__(self, config_path: Optional[str] = None, logger: Optional[Logger] = None, max_workers: int = 2, max_process_count: Optional[int] = None):
        """バッチ処理の初期設定"""
        load_dotenv()
        self.project_root = os.getenv('PROJECT_ROOT', os.getcwd())
        self.logger = logger if logger else Logger(logger_name=self.__class__.__name__).get_logger()
        self.max_workers = max_workers
        self.config_path = config_path or self.default_config["config_path"]
        self.config = self.default_config.copy()
        self.start_time = None
        self.end_time = None
        self.max_process_count = max_process_count  # 処理件数制限
        self.processed_count = 0  # 処理件数のカウンタ

        self.hooks: Dict[HookType, List[Tuple[int, Callable[[], None]]]] = {
            hook_type: [] for hook_type in HookType
        }

        self.load_config(self.config_path)
        self._start_config_watcher(self.config_path)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def load_config(self, config_path: str) -> None:
        """コンフィグファイルを読み込む"""
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}. Using default settings.")

            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r') as config_file:
                    loaded_config = yaml.safe_load(config_file)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as config_file:
                    loaded_config = json.load(config_file)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")

            self.validate_config(loaded_config)
            self.config.update(loaded_config)
            self.logger.info(f"Configuration loaded from {config_path}.")
        
        except Exception as e:
            self.handle_error(f"Error loading configuration from '{config_path}': {e}")

    def validate_config(self, config: Dict) -> None:
        """読み込んだコンフィグのバリデーション"""
        required_keys = ["setting_1", "setting_2"]
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise ValueError(f"Missing required config keys: {', '.join(missing_keys)}")

    def reload_config(self, config_path: Optional[str] = None) -> None:
        """コンフィグの再読み込み"""
        self.logger.info("Reloading configuration.")
        if config_path:
            self.config_path = config_path
        self.load_config(self.config_path)

    def _start_config_watcher(self, config_path: str) -> None:
        """Configファイルの変更を監視するためのウォッチャーを開始"""
        event_handler = ConfigChangeHandler(self)
        observer = Observer()
        observer.schedule(event_handler, path=os.path.dirname(config_path), recursive=False)
        observer.start()
        self.logger.info(f"Started watching for changes in {config_path}.")

    def execute(self) -> None:
        """バッチ処理のメインエントリポイント"""
        self.start_time = time.time()
        self.logger.info("Batch process started.")
        errors = []

        try:
            self._execute_phase(HookType.PRE_SETUP, HookType.POST_SETUP, self.setup, errors)
            self._execute_phase(HookType.PRE_PROCESS, HookType.POST_PROCESS, self.process, errors)
        finally:
            self._execute_phase(HookType.PRE_CLEANUP, HookType.POST_CLEANUP, self.cleanup, errors)
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.logger.info(f"Batch process completed in {duration:.2f} seconds.")
            
            if errors:
                self.handle_error(f"Batch process encountered errors: {errors}", raise_exception=True)

    def _execute_phase(self, pre_hook_type: HookType, post_hook_type: HookType, phase_function: Callable[[], None], errors: List[Exception]) -> None:
        """各フェーズの前後フックとフェーズ自体の実行を管理"""
        phase_name = pre_hook_type.name.split('_')[1].lower()
        self._log_phase_start(phase_name)
        phase_start_time = time.time()

        try:
            self._execute_hooks(pre_hook_type)
            phase_function()  # フェーズ（setup, process, cleanup）の実行
            self._execute_hooks(post_hook_type)
        except Exception as e:
            errors.append(e)
            self.logger.error(f"{phase_name.capitalize()} phase encountered an error: {e}")
        finally:
            phase_duration = time.time() - phase_start_time
            self._log_phase_end(phase_name, phase_duration)

    def process(self) -> None:
        """メイン処理フェーズで行う処理。サブクラスで実装"""
        while self.max_process_count is None or self.processed_count < self.max_process_count:
            try:
                # サブクラスで具体的な処理を実装
                self.run_task()
                self.processed_count += 1  # 処理件数の更新
            except Exception as e:
                self.logger.error(f"Error processing task: {e}")
                break

    def _log_phase_start(self, phase_name: str) -> None:
        """フェーズ開始時のログ出力"""
        self.logger.info(f"Starting {phase_name} phase.")
    
    def _log_phase_end(self, phase_name: str, duration: float) -> None:
        """フェーズ終了時のログ出力"""
        self.logger.info(f"{phase_name.capitalize()} phase completed in {duration:.2f} seconds.")

    def add_hook(self, hook_type: HookType, func: Callable[[], None], priority: int = 0) -> None:
        """優先度を持つフックを追加"""
        hook_list = self.hooks[hook_type]
        for index, (existing_priority, _) in enumerate(hook_list):
            if existing_priority == priority:
                self.logger.warning(f"Hook already exists in {hook_type.value} with priority {priority}: {func}")
                return
            elif existing_priority < priority:
                hook_list.insert(index, (priority, func))
                break
        else:
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
                    self.logger.error(f"Hook execution encountered an error: {e}")
                    errors.append(e)

        if errors:
            self.handle_error(f"Errors encountered during hook execution: {errors}")

    def handle_error(self, message: str, raise_exception: bool = False) -> None:
        """エラーハンドリング"""
        self.logger.error(message)
        if raise_exception:
            raise RuntimeError(message)

    def _handle_shutdown(self, signum, frame):
        """シグナルを受け取って安全にシャットダウンする"""
        self.logger.info(f"Received shutdown signal {signum}. Initiating cleanup.")
        self.cleanup()
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.logger.info(f"Batch process terminated in {duration:.2f} seconds.")
        exit(0)

    @abstractmethod
    def setup(self) -> None:
        """セットアップフェーズ。サブクラスで実装"""
        pass

    @abstractmethod
    def run_task(self) -> None:
        """個別のタスクを実行。サブクラスで実装"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """クリーンアップフェーズ。サブクラスで実装"""
        pass
