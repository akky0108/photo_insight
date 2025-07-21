import os
import time
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Optional, Callable, List, Dict, Any
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
        logger: Optional[logging.Logger] = None,
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
        self.max_workers = max(1, max_workers or 0)

        # 明示的にconfig_pathを解決
        resolved_config_path = (
            config_path
            if config_path is not None
            else os.path.join(self.project_root, "config", "config.yaml")
        )
        self.config_path = resolved_config_path

        # 1) loggerが外部から渡されていればそれを使い、なければConfigManagerで取得する
        if logger is None:
            # config_managerが渡されている場合はそれを使う
            if config_manager is None:
                config_manager = ConfigManager(config_path=self.config_path)
            logger = config_manager.get_logger(self.__class__.__name__)
        else:
            # loggerが渡されたならconfig_managerはなければ生成（logger渡しなし）
            if config_manager is None:
                config_manager = ConfigManager(config_path=self.config_path)

        self.logger = logger
        self.config_manager = config_manager

        # HookManagerはloggerを明示的に渡すよう変更
        if hook_manager is None:
            hook_manager = HookManager(max_workers=self.max_workers, logger=self.logger)
        else:
            hook_manager.logger = self.logger

        self.hook_manager = hook_manager

        # SignalHandlerもloggerを渡す（signal_handler生成時にlogger渡しが無ければ明示的に設定）
        if signal_handler is None:
            signal_handler = SignalHandler(self)
        signal_handler.logger = self.logger
        self.signal_handler = signal_handler

        self.processed_count = 0
        self.config = self.config_manager.config
        self._lock = Lock()

    def execute(self, *args, **kwargs) -> None:
        """
        バッチ処理のメイン実行メソッド。
        setup→process→cleanup の各フェーズのフックと処理を順次実行し、
        処理時間計測、エラーハンドリングを行う。

        Raises:
            RuntimeError: フェーズ中にエラーが起きた場合に送出される
        """
        self.start_time = time.time()
        self.logger.info(f"[{self.__class__.__name__}] Batch process started.")
        errors = []

        try:
            self._execute_phase(
                HookType.PRE_SETUP, HookType.POST_SETUP, self.setup, errors
            )
            self._execute_phase(
                HookType.PRE_PROCESS, HookType.POST_PROCESS, lambda: self.process(*args, **kwargs), errors
            )
        finally:
            self._execute_phase(
                HookType.PRE_CLEANUP, HookType.POST_CLEANUP, self.cleanup, errors
            )
            duration = time.time() - self.start_time
            self.logger.info(f"[{self.__class__.__name__}] Batch process completed in {duration:.2f} seconds.")

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
            self.logger.error(f"[{self.__class__.__name__}] Error in {hook_type.name} hooks: {e}")
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
            self.logger.info(f"[{self.__class__.__name__}] Executing {name} phase.")
            func()
            duration = time.time() - start_time
            self.logger.info(
                f"[{self.__class__.__name__}] {name.capitalize()} phase completed in {duration:.2f} seconds."
                )
        except Exception as e:
            self.logger.error(f"[{self.__class__.__name__}] Error during {name} phase: {e}")
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
        self.logger.info(f"[{self.__class__.__name__}] Executing common setup tasks.")
        self.data = self.get_data()

    def process(self, data: Optional[List[Dict]] = None) -> None:
        self.logger.info(f"[{self.__class__.__name__}] Executing common batch processing tasks.")
        if data is None:
            data = self.get_data()

        batches = self._generate_batches(data)
        failed_batches = []
        all_results: List[Dict[str, Any]] = []

        if not batches:
            self.logger.info("No data to process. Skipping batch execution.")
            return

        futures = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i, batch in enumerate(batches):
                if self._should_stop_processing():
                    self.logger.warning(
                        f"[{self.__class__.__name__}] Stopping before batch {i + 1} due to interrupt condition."
                    )
                    break

                future = executor.submit(self._safe_process_batch, batch, self.get_lock())
                futures[future] = (i + 1, batch)

            for future in as_completed(futures):
                batch_idx, batch_data = futures[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except Exception as e:
                    self.logger.error(
                        f"[{self.__class__.__name__}] [Batch {batch_idx}] Failed in thread: {e}",
                        exc_info=True,
                    )
                    # 失敗バッチのトランケートされた内容をDEBUGログで出力
                    truncated = str(batch_data)[:100]  # 100文字まで表示（必要に応じて変更）
                    self.logger.debug(
                        f"[{self.__class__.__name__}] [Batch {batch_idx}] Failed batch data (truncated): {truncated}"
                    )
                    failed_batches.append(batch_idx)

        # 処理統計（例：成功数、失敗数、平均スコアなど）
        summary = self._summarize_results(all_results)
        self.logger.info(f"[{self.__class__.__name__}] Batch Summary: {summary}")

        if self._should_log_summary_detail():
            self.logger.info(
                f"Processed {summary['total']} items. "
                f"Success: {summary['success']}, Failures: {summary['failure']}"
            )
            if summary["avg_score"] is not None:
                self.logger.info(f"Average score: {summary['avg_score']}")

        self.final_summary = summary

        if failed_batches:
            self.logger.warning(
                f"[{self.__class__.__name__}] Processing completed with failures in batches: {failed_batches}"
            )
        else:
            self.logger.info(f"[{self.__class__.__name__}] All batches processed successfully.")
            if hasattr(self, "completed_all_batches"):
                self.completed_all_batches = True

    def _safe_process_batch(self, batch: List[Dict], lock: Lock) -> List[Dict[str, Any]]:
        """
        スレッドセーフなバッチ処理。必要ならファイル書き込み時にlock使用。
        Returns:
            List[Dict[str, Any]]: 各ファイル/データに対する処理結果
        """
        return self._process_batch(batch)

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        処理結果のリストから成功/失敗件数と平均スコアを集計する共通メソッド。

        Args:
            results (List[Dict[str, Any]]): 各データ処理の結果

        Returns:
            Dict[str, Any]: 集計サマリー（total, success, failure, avg_score）
        """
        success = [r for r in results if r.get("status") == "success"]
        failure = [r for r in results if r.get("status") != "success"]
        avg_score = (
            sum(r.get("score", 0) for r in success) / len(success)
            if success else None
        )

        return {
            "total": len(results),
            "success": len(success),
            "failure": len(failure),
            "avg_score": round(avg_score, 2) if avg_score is not None else None,
        }

    def cleanup(self) -> None:
        """
        クリーンアップフェーズの共通処理。必要に応じてサブクラスでオーバーライド可能。
        """
        self.logger.info(f"[{self.__class__.__name__}] Executing common cleanup tasks.")

    def _generate_batches(self, data: List[Dict], batch_size: Optional[int] = None) -> List[List[Dict]]:
        """
        データを指定サイズのバッチに分割する汎用メソッド。

        Args:
            data (List[Dict]): 分割対象データ（各要素は辞書）
            batch_size (Optional[int]): 明示的なバッチサイズを指定する場合に使用。
                                        指定がない場合は max_workers をもとに自動計算。

        Returns:
            List[List[Dict]]: 分割されたバッチのリスト（各バッチは List[Dict]）

        Note:
            サブクラスでバッチの意味論（例：ディレクトリ単位、グループ単位）を持つ場合は
            本メソッドをオーバーライドしてカスタマイズすること。
        """
        if not data:
            return []

        # 引数 > self.batch_size > 自動スレッド分割
        batch_size = (
            batch_size
            or getattr(self, "batch_size", None)
            or max(1, len(data) // self.max_workers)
        )

        # バッチサイズが未指定の場合、自動的にスレッド数に応じて割り当て
        if batch_size is None:
            batch_size = max(1, len(data) // self.max_workers)

        # 指定されたバッチサイズでスライス分割
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def _should_stop_processing(self) -> bool:
        """
        サブクラスが定義した中断条件があれば中断する。

        例: memory_threshold_exceeded など。
        デフォルトでは存在チェックだけで動作。
        """
        return getattr(self, "memory_threshold_exceeded", False)

    def _should_log_summary_detail(self) -> bool:
        """
        詳細なサマリーログ出力を行うかどうかを判定する。

        ・環境変数 DEBUG_LOG_SUMMARY=1 が設定されていれば True
        ・設定ファイル内 config["debug"]["log_summary_detail"] == True でも有効
        """
        return (
            os.getenv("DEBUG_LOG_SUMMARY") == "1"
            or self.config.get("debug", {}).get("log_summary_detail", False)
        )

    def get_lock(self) -> Lock:
        """スレッドセーフな処理に使うロックを提供する。"""
        return self._lock

    @abstractmethod
    def get_data(self, *args, **kwargs) -> List[Dict]:
        """
        データ取得メソッド。具体的な実装はサブクラスで行う必要がある。

        Returns:
            List[Dict]: バッチ処理対象のデータ一覧
        """
        pass

    @abstractmethod
    def _process_batch(self, batch: List[Dict]) -> List[Dict[str, Any]]:
        """
        バッチ単位の処理メソッド。サブクラスでの実装が必須。

        Args:
            batch (List[Dict]): 単一バッチのデータ

        Returns:
            List[Dict[str, Any]]: 各データごとの処理結果を返す（例：status/score/エラーなど）
        """
        pass