from abc import ABC, abstractmethod
import time
import json
import logging
from typing import Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor
from log_util import Logger  # 修正したカスタムロガーのインポート
from enum import Enum

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
        self.logger: Optional[Logger] = logger or logging.getLogger('BatchProcessLogger')
        
        # 最大ワーカー数の設定
        self.max_workers = max_workers

        # フックリストの初期化
        self.pre_setup_hooks: List[Callable[[], None]] = []  # セットアップ前のフック
        self.post_setup_hooks: List[Callable[[], None]] = []  # セットアップ後のフック
        self.pre_process_hooks: List[Callable[[], None]] = []  # プロセス開始前のフック
        self.post_process_hooks: List[Callable[[], None]] = []  # プロセス終了後のフック
        self.pre_cleanup_hooks: List[Callable[[], None]] = []  # クリーンアップ前のフック
        self.post_cleanup_hooks: List[Callable[[], None]] = []  # クリーンアップ後のフック

        # 設定ファイルパスの保持
        self.config_path = config_path  
        # デフォルト設定をコピー
        self.config = self.default_config.copy()
        # 設定ファイルが指定されていれば読み込み
        if config_path:
            self.load_config(config_path)

        # 開始時間と終了時間の初期化
        self.start_time = None
        self.end_time = None

    def load_config(self, config_path: str) -> None:
        try:
            with open(config_path, 'r') as config_file:
                loaded_config = json.load(config_file)
            self.config.update(loaded_config)
            self.logger.info(f"Configuration loaded from {config_path}.")
        except FileNotFoundError:
            self.logger.warning(f"Config file not found at {config_path}. Using default settings.")
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON config file '{config_path}': {e}. Using default settings.")
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
            self._setup_phase(errors)
            self._process_phase(errors)
        finally:
            self._cleanup_phase(errors)
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.logger.info(f"Batch process completed in {duration:.2f} seconds.")
            
            if errors:
                error_messages = "\n".join(str(e) for e in errors)
                self.logger.error(f"Batch process completed with errors: {error_messages}")
                raise RuntimeError(f"Batch process encountered errors: {error_messages}")

    def _setup_phase(self, errors: List[Exception]) -> None:
        """Setupフェーズを実行する。"""
        self._execute_phase("setup", self.setup, self.pre_setup_hooks, self.post_setup_hooks, errors)

    def _process_phase(self, errors: List[Exception]) -> None:
        """Processフェーズを実行する。"""
        self._execute_phase("process", self.process, self.pre_process_hooks, self.post_process_hooks, errors)

    def _cleanup_phase(self, errors: List[Exception]) -> None:
        """Cleanupフェーズを実行する。"""
        self._execute_phase("cleanup", self.cleanup, self.pre_cleanup_hooks, self.post_cleanup_hooks, errors)

    def _execute_phase(
        self,
        phase_name: str,
        phase_function: Callable[[], None],
        pre_hooks: List[Optional[Callable[[], None]]],
        post_hooks: List[Optional[Callable[[], None]]],
        errors: List[Exception]
    ) -> None:
        self._log_phase_start(phase_name)
        phase_start_time = time.time()
        
        try:
            # 必要に応じて並列または順次でフックを実行する
            if self.max_workers > 1:
                self._execute_hooks_parallel(pre_hooks)
            else:
                self._execute_hooks_sequential(pre_hooks)

            phase_function()

            if self.max_workers > 1:
                self._execute_hooks_parallel(post_hooks)
            else:
                self._execute_hooks_sequential(post_hooks)
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
        hook_list = self._get_hook_list(hook_type)
        if func in hook_list:
            self.logger.warning(f"Hook already exists in {hook_type.value}: {func}")
        else:
            hook_list.append(func)

    def _get_hook_list(self, hook_type: HookType) -> List[Callable[[], None]]:
        """フックタイプに応じて対応するフックリストを返す"""
        if hook_type == HookType.PRE_SETUP:
            return self.pre_setup_hooks
        elif hook_type == HookType.POST_SETUP:
            return self.post_setup_hooks
        elif hook_type == HookType.PRE_PROCESS:
            return self.pre_process_hooks
        elif hook_type == HookType.POST_PROCESS:
            return self.post_process_hooks
        elif hook_type == HookType.PRE_CLEANUP:
            return self.pre_cleanup_hooks
        elif hook_type == HookType.POST_CLEANUP:
            return self.post_cleanup_hooks
        else:
            raise ValueError(f"無効なフックタイプです: {hook_type}")

    def _execute_hooks_parallel(self, hooks: List[Optional[Callable[[], None]]]) -> None:
        """フックリストに含まれる各フックを並列で実行する。"""
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

    def _execute_hooks_sequential(self, hooks: List[Optional[Callable[[], None]]]) -> None:
        """フックリストに含まれる各フックを順次で実行する。"""
        errors = []
        for hook in hooks:
            if hook:
                try:
                    hook()
                except Exception as e:
                    self.logger.error(f"Hook execution encountered an error: {e}")
                    errors.append(e)
        
        if errors:
            raise RuntimeError(f"Errors encountered during hook execution: {errors}")
