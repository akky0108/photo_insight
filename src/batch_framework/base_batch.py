from abc import ABC, abstractmethod
import time
import json
import logging
from typing import Optional, Callable, List
from concurrent.futures import ThreadPoolExecutor
from log_util import Logger  # 修正したカスタムロガーのインポート

class BaseBatchProcessor(ABC):
    # デフォルト設定をクラス属性として定義
    default_config = {"setting_1": "default_value_1", "setting_2": "default_value_2"}

    def __init__(self, config_path: Optional[str] = None, logger: Optional[Logger] = None, max_workers: int = 2):
        """
        バッチ処理の基礎クラス。共通のセットアップ、プロセス、クリーンアップロジックを提供。
        
        :param config_path: JSON形式の設定ファイルのパス
        :param logger: ロギング用のカスタムロガー
        :param max_workers: 並列処理時のスレッド数（デフォルトは3）
        """
        # Loggerの設定。指定がない場合は標準のlogging.Loggerを使用
        self.logger = logger or logging.getLogger('BatchProcessLogger')
        
        # 設定の読み込み
        self.config = self.default_config.copy()
        if config_path:
            self.load_config(config_path)

        self.start_time = None
        self.end_time = None
        self.max_workers = max_workers

        # フックポイントリストの初期化
        self.pre_setup_hooks: List[Callable[[], None]] = [self.pre_setup]
        self.post_setup_hooks: List[Callable[[], None]] = [self.post_setup]
        self.pre_process_hooks: List[Callable[[], None]] = [self.pre_process]
        self.post_process_hooks: List[Callable[[], None]] = [self.post_process]
        self.pre_cleanup_hooks: List[Callable[[], None]] = [self.pre_cleanup]
        self.post_cleanup_hooks: List[Callable[[], None]] = [self.post_cleanup]

    def load_config(self, config_path: str) -> None:
        """
        指定されたパスから設定ファイルを読み込み、デフォルト設定とマージする。

        :param config_path: 設定ファイルのパス
        """
        try:
            with open(config_path, 'r') as config_file:
                loaded_config = json.load(config_file)
            self.config.update(loaded_config)  # デフォルト設定に上書き
            self.logger.info(f"Configuration loaded from {config_path}.")
        except FileNotFoundError:
            self.logger.warning(f"Config file not found at {config_path}. Using default settings.")
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON config file '{config_path}': {e}. Using default settings.")
        except Exception as e:
            self.logger.error(f"Unexpected error loading configuration from '{config_path}': {e}. Using default settings.")

    def register_hook(self, phase_name: str, hook: Callable[[], None]) -> None:
        """
        指定したフェーズにカスタムフックを登録するメソッド。

        :param phase_name: フェーズ名（pre_setup, post_setup, pre_process, post_process, pre_cleanup, post_cleanup）
        :param hook: 実行するフック関数
        """
        hook_list_name = f"{phase_name}_hooks"
        if hasattr(self, hook_list_name):
            getattr(self, hook_list_name).append(hook)
            self.logger.info(f"Hook registered to {phase_name} phase.")
        else:
            self.logger.warning(f"Invalid phase name: {phase_name}")

    @abstractmethod
    def setup(self) -> None:
        """リソースのセットアップを行う抽象メソッド。継承先で実装する必要がある。"""
        pass
    
    @abstractmethod
    def process(self) -> None:
        """メインの処理を行う抽象メソッド。継承先で実装する必要がある。"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """リソースのクリーンアップを行う抽象メソッド。継承先で実装する必要がある。"""
        pass

    def pre_setup(self) -> None:
        """Setupフェーズの前に実行されるフックポイント"""
        self.logger.info("Running pre-setup hook.")

    def post_setup(self) -> None:
        """Setupフェーズの後に実行されるフックポイント"""
        self.logger.info("Running post-setup hook.")

    def pre_process(self) -> None:
        """Processフェーズの前に実行されるフックポイント"""
        self.logger.info("Running pre-process hook.")

    def post_process(self) -> None:
        """Processフェーズの後に実行されるフックポイント"""
        self.logger.info("Running post-process hook.")

    def pre_cleanup(self) -> None:
        """Cleanupフェーズの前に実行されるフックポイント"""
        self.logger.info("Running pre-cleanup hook.")

    def post_cleanup(self) -> None:
        """Cleanupフェーズの後に実行されるフックポイント"""
        self.logger.info("Running post-cleanup hook.")
    
    def execute(self) -> None:
        """
        バッチプロセスを順番に実行するメインメソッド。
        各フェーズの実行時間を計測し、ログに記録する。
        """
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
        """
        各フェーズを実行し、その時間を計測してログに出力する。フックを使用する。

        :param phase_name: フェーズ名
        :param phase_function: フェーズのメイン関数
        :param pre_hooks: フェーズ前に実行されるフックリスト
        :param post_hooks: フェーズ後に実行されるフックリスト
        :param errors: 発生したエラーのリスト
        """
        self._log_phase_start(phase_name)
        phase_start_time = time.time()
        
        try:
            self._execute_hooks_parallel(pre_hooks)
            phase_function()
            self._execute_hooks_parallel(post_hooks)
        except Exception as e:
            errors.append(e)
            self.logger.error(f"{phase_name.capitalize()} phase encountered a general error: {e}")
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
