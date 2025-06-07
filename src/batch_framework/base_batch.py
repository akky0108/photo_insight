import os
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Optional, Callable, List, Dict
from watchdog.events import FileSystemEventHandler

from batch_framework.core.hook_manager import HookManager, HookType
from batch_framework.core.config_manager import ConfigManager
from batch_framework.core.signal_handler import SignalHandler


class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, processor: "BaseBatchProcessor"):
        """
        コンフィグファイルの変更を監視し、変更があればプロセッサに通知するハンドラクラス

        Args:
            processor (BaseBatchProcessor): 監視対象のバッチプロセッサ
        """
        self.processor = processor
        self._last_modified_time = 0

    def on_modified(self, event):
        """
        ファイルが変更された際に呼ばれるメソッド。短時間で連続変更された場合は無視する。

        Args:
            event: watchdogのファイル変更イベント
        """
        now = time.time()
        if now - self._last_modified_time < 1.0:
            return
        self._last_modified_time = now
        if event.src_path == self.processor.config_path:
            self.processor.logger.info(
                f"Config file {event.src_path} has been modified. Reloading..."
            )
            self.processor.reload_config()


class BaseBatchProcessor(ABC):
    def __init__(
        self,
        config_path: Optional[str] = None,
        max_workers: int = 2,
        hook_manager: Optional[HookManager] = None,
        config_manager: Optional[ConfigManager] = None,
        signal_handler: Optional[SignalHandler] = None,
    ):
        """
        バッチ処理の基底クラスコンストラクタ。
        設定ファイルのロード、フック管理、設定監視の初期化を行う。

        Args:
            config_path (Optional[str]): 設定ファイルのパス。指定なければデフォルトパスを使用。
            max_workers (int): フックの並列実行数。
        """
        load_dotenv()
        self.project_root = os.getenv("PROJECT_ROOT") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.max_workers = max_workers

        # 明示的にconfig_pathを解決
        resolved_config_path = (
            config_path
            if config_path is not None
            else os.path.join(self.project_root, "config", "config.yaml")
        )

        # 明示的に依存性注入する
        self.config_path = resolved_config_path
        self.config_manager = (
            config_manager
            if config_manager is not None
            else ConfigManager(config_path=self.config_path)
        )
        self.logger = self.config_manager.get_logger("BaseBatchProcessor")

        self.hook_manager = (
            hook_manager
            if hook_manager is not None
            else HookManager(max_workers=self.max_workers)
        )
        self.hook_manager.logger = self.logger

        self.signal_handler = (
            signal_handler if signal_handler is not None else SignalHandler(self)
        )
        self.signal_handler.logger = self.logger

        self.processed_count = 0
        self.config = self.config_manager.config

    def execute(self) -> None:
        """
        バッチ処理のメイン実行メソッド。
        setup→process→cleanup の各フェーズのフックと処理を順次実行し、
        処理時間計測、エラーハンドリングを行う。

        Raises:
            RuntimeError: フェーズ中にエラーが起きた場合に送出される
        """
        self.start_time = time.time()
        self.logger.info("Batch process started.")
        errors = []

        try:
            self._execute_phase(
                HookType.PRE_SETUP, HookType.POST_SETUP, self.setup, errors
            )
            self._execute_phase(
                HookType.PRE_PROCESS, HookType.POST_PROCESS, self.process, errors
            )
        finally:
            self._execute_phase(
                HookType.PRE_CLEANUP, HookType.POST_CLEANUP, self.cleanup, errors
            )
            duration = time.time() - self.start_time
            self.logger.info(f"Batch process completed in {duration:.2f} seconds.")

            if errors:
                self.handle_error(
                    f"Batch process encountered errors: {errors}", raise_exception=True
                )

    def _execute_phase(
        self,
        pre_hook_type: HookType,
        post_hook_type: HookType,
        phase_function: Callable[[], None],
        errors: List[Exception],
    ) -> None:
        """
        フェーズごとに、事前フック、メイン関数、事後フックを順に実行する。

        Args:
            pre_hook_type (HookType): 事前フックの種類
            post_hook_type (HookType): 事後フックの種類
            phase_function (Callable[[], None]): フェーズ本体の関数
            errors (List[Exception]): 発生した例外を格納するリスト
        """
        phase_name = pre_hook_type.name.split("_")[1].lower()
        self._run_phase_hooks(pre_hook_type, errors)
        self._run_phase_function(phase_function, phase_name, errors)
        self._run_phase_hooks(post_hook_type, errors)

    def _run_phase_hooks(self, hook_type: HookType, errors: List[Exception]) -> None:
        """
        指定フックタイプのフック関数群を実行し、例外を捕捉してerrorsに格納する。

        Args:
            hook_type (HookType): 実行するフックタイプ
            errors (List[Exception]): 例外格納用リスト
        """
        try:
            self.hook_manager.execute_hooks(hook_type)
        except Exception as e:
            self.logger.error(f"Error in {hook_type.name} hooks: {e}")
            errors.append(e)

    def _run_phase_function(
        self, func: Callable[[], None], name: str, errors: List[Exception]
    ) -> None:
        """
        フェーズ本体の関数を実行し、例外を捕捉してerrorsに格納する。

        Args:
            func (Callable[[], None]): 実行する関数
            name (str): フェーズ名（ログ用）
            errors (List[Exception]): 例外格納用リスト
        """
        try:
            start_time = time.time()
            self.logger.info(f"Executing {name} phase.")
            func()
            duration = time.time() - start_time
            self.logger.info(
                f"{name.capitalize()} phase completed in {duration:.2f} seconds."
            )
        except Exception as e:
            self.logger.error(f"Error during {name} phase: {e}")
            errors.append(e)

    def add_hook(
        self, hook_type: HookType, func: Callable[[], None], priority: int = 0
    ) -> None:
        """
        指定された種類のフックに関数を登録する。
        """
        self.hook_manager.add_hook(hook_type, func, priority)

    def reload_config(self, config_path: Optional[str] = None) -> None:
        """
        設定ファイルを再読み込みする。
        """
        self.config_manager.reload_config(config_path)
        self.config = self.config_manager.config

    def handle_error(self, message: str, raise_exception: bool = False) -> None:
        """
        エラー処理：ログ出力し、必要に応じて例外をスローする。
        """
        self.logger.error(message)
        if raise_exception:
            raise RuntimeError(message)

    def setup(self) -> None:
        """
        セットアップフェーズの共通処理。必要に応じてサブクラスでオーバーライド可能。
        """
        self.logger.info("Executing common setup tasks in BaseBatchProcessor.")

    def process(self) -> None:
        """
        メイン処理フェーズ。データ取得とバッチ単位での処理を行う。
        """
        self.logger.info(
            "Executing common batch processing tasks in BaseBatchProcessor."
        )
        data = self.get_data()
        batches = self._generate_batches(data)
        for i, batch in enumerate(batches):
            self.logger.info(f"Processing batch {i + 1}...")
            self._process_batch(batch)

    def cleanup(self) -> None:
        """
        クリーンアップフェーズの共通処理。必要に応じてサブクラスでオーバーライド可能。
        """
        self.logger.info("Executing common cleanup tasks in BaseBatchProcessor.")

    def _generate_batches(self, data: List[Dict]) -> List[List[Dict]]:
        """
        データを設定されたサイズで分割し、バッチ一覧を生成する。

        Args:
            data (List[Dict]): 元データ

        Returns:
            List[List[Dict]]: 分割されたバッチリスト
        """
        batch_size = self.config.get("batch_size", 100)
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    @abstractmethod
    def get_data(self) -> List[Dict]:
        """
        データ取得メソッド。具体的な実装はサブクラスで行う必要がある。

        Returns:
            List[Dict]: バッチ処理対象のデータ一覧
        """
        pass

    @abstractmethod
    def _process_batch(self, batch: List[Dict]) -> None:
        """
        バッチ単位の処理メソッド。サブクラスでの実装が必須。

        Args:
            batch (List[Dict]): 単一バッチのデータ
        """
        pass
