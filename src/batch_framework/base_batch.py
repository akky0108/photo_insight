import yaml
import json
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict
from concurrent.futures import ThreadPoolExecutor
from log_util import Logger
from enum import Enum
import time

# フックタイプを定義する列挙型
class HookType(Enum):
    PRE_SETUP = 'pre_setup'
    POST_SETUP = 'post_setup'
    PRE_PROCESS = 'pre_process'
    POST_PROCESS = 'post_process'
    PRE_CLEANUP = 'pre_cleanup'
    POST_CLEANUP = 'post_cleanup'

class BaseBatchProcessor(ABC):
    # デフォルト設定をクラス属性として定義
    default_config = {"setting_1": "default_value_1", "setting_2": "default_value_2"}

    def __init__(self, config_path: Optional[str] = None, logger: Optional[Logger] = None, max_workers: int = 2):
        # ロガーの初期化。提供されたロガーがない場合はデフォルトのロガーを使用
        self.logger: Logger = logger if logger else Logger(logger_name='BaseBatchProcessor').get_logger()
        
        # 最大ワーカー数の設定
        self.max_workers = max_workers

        # フックリストの初期化を辞書にまとめる
        self.hooks: Dict[HookType, List[Callable[[], None]]] = {
            hook_type: [] for hook_type in HookType
        }

        # 設定ファイルパスの保持
        self.config_path = config_path  
        self.config = self.default_config.copy()
        if config_path:
            self.load_config(config_path)

        # 開始時間と終了時間の初期化
        self.start_time = None
        self.end_time = None

    def load_config(self, config_path: str) -> None:
        try:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r') as config_file:
                    loaded_config = yaml.safe_load(config_file)
            elif config_path.endswith('.json'):
                with open(config_path, 'r') as config_file:
                    loaded_config = json.load(config_file)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")
            
            self.config.update(loaded_config)
            self.logger.info(f"Configuration loaded from {config_path}.")
        except FileNotFoundError:
            self.logger.warning(f"Config file not found at {config_path}. Using default settings.")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error(f"Error decoding config file '{config_path}': {e}. Using default settings.")
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration from '{config_path}': {e}. Using default settings.")

    def reload_config(self, config_path: Optional[str] = None) -> None:
        """設定を動的にリロードする。"""
        self.logger.info("Reloading configuration.")
        if config_path:
            self.load_config(config_path)
            self.config_path = config_path  # 新しいパスを保持
        elif self.config_path:
            self.load_config(self.config_path)  # 保存されたパスを使用
        else:
            self.logger.warning("No configuration path provided for reloading.")

    def execute(self) -> None:
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
                error_messages = "\n".join(str(e) for e in errors)
                self.logger.error(f"Batch process completed with errors: {error_messages}")
                raise RuntimeError(f"Batch process encountered errors: {error_messages}")

    def _execute_phase(
        self,
        pre_hook_type: HookType,
        post_hook_type: HookType,
        phase_function: Callable[[], None],
        errors: List[Exception]
    ) -> None:
        """各フェーズの実行とフックの処理"""
        phase_name = pre_hook_type.name.split('_')[1].lower()
        self._log_phase_start(phase_name)
        phase_start_time = time.time()

        try:
            self._execute_hooks(pre_hook_type)
            phase_function()
            self._execute_hooks(post_hook_type)
        except Exception as e:
            errors.append(e)
            self.logger.error(f"{phase_name.capitalize()} phase encountered an error: {e}")
            raise
        finally:
            phase_duration = time.time() - phase_start_time
            self._log_phase_end(phase_name, phase_duration)

    def _log_phase_start(self, phase_name: str) -> None:
        """フェーズの開始をログに記録する。"""
        self.logger.info(f"Starting {phase_name} phase.")
    
    def _log_phase_end(self, phase_name: str, duration: float) -> None:
        """フェーズの終了をログに記録する。"""
        self.logger.info(f"{phase_name.capitalize()} phase completed in {duration:.2f} seconds.")

    def add_hook(self, hook_type: HookType, func: Callable[[], None]) -> None:
        """フックを追加する"""
        hook_list = self.hooks[hook_type]
        if func in hook_list:
            self.logger.warning(f"Hook already exists in {hook_type.value}: {func}")
        else:
            hook_list.append(func)

    def _execute_hooks(self, hook_type: HookType) -> None:
        """フックリストに含まれる各フックを実行する。"""
        hooks = self.hooks[hook_type]
        errors = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(hook) for hook in hooks if hook]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Hook execution encountered an error: {e}")
                    errors.append(e)
        
        if errors:
            raise RuntimeError(f"Errors encountered during hook execution: {errors}")
    
    @abstractmethod
    def setup(self) -> None:
        pass
    
    @abstractmethod
    def process(self) -> None:
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        pass
