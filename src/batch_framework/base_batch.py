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
    PRE_SETUP = 'pre_setup'
    POST_SETUP = 'post_setup'
    PRE_PROCESS = 'pre_process'
    POST_PROCESS = 'post_process'
    PRE_CLEANUP = 'pre_cleanup'
    POST_CLEANUP = 'post_cleanup'


class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, processor: 'BaseBatchProcessor'):
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
            self.processor.logger.info(f"Config file {event.src_path} has been modified. Reloading...")
            self.processor.reload_config()


class BaseBatchProcessor(ABC):
    def __init__(self, config_path: Optional[str] = None, max_workers: int = 2):
        """
        バッチ処理の基底クラスコンストラクタ。
        設定ファイルのロード、フック管理、設定監視の初期化を行う。

        Args:
            config_path (Optional[str]): 設定ファイルのパス。指定なければデフォルトパスを使用。
            max_workers (int): フックの並列実行数。
        """
        load_dotenv()
        self.project_root = os.getenv('PROJECT_ROOT') or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.default_config = {
            "config_path": os.path.join(self.project_root, "config", "config.yaml"),
            "batch_size": 100
        }

        self.logger = self._get_default_logger()
        self.max_workers = max_workers
        self.config_path = config_path or self.default_config["config_path"]
        self.config = self.default_config.copy()
        self.processed_count = 0

        # フック管理辞書（HookTypeごとに優先度付きの関数リスト）
        self.hooks: Dict[HookType, List[Tuple[int, Callable[[], None]]]] = {hook_type: [] for hook_type in HookType}

        self.load_config(self.config_path)
        self._start_config_watcher(self.config_path)

        # プロセス終了シグナルのハンドリング登録
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _get_default_logger(self):
        """
        デフォルトのロガーを取得する。

        Returns:
            logging.Logger: ロガーオブジェクト
        """
        from utils.app_logger import Logger
        return Logger(
            project_root=self.project_root,
            logger_name=self.__class__.__name__
        ).get_logger()

    def load_config(self, config_path: str) -> None:
        """
        設定ファイルを読み込み、self.configに反映する。

        Args:
            config_path (str): 設定ファイルのパス

        Raises:
            Exception: 読み込み失敗時に例外を送出する
        """
        try:
            self.logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                self.config.clear()
                self.config.update(yaml.safe_load(f))
        except Exception:
            self.logger.exception("Failed to load configuration.")
            raise

    def reload_config(self, config_path: Optional[str] = None) -> None:
        """
        設定ファイルを再読み込みする。

        Args:
            config_path (Optional[str]): 新たな設定ファイルパス。Noneなら既存のself.config_pathを使う。
        """
        self.logger.info("Reloading configuration.")
        if config_path:
            self.config_path = config_path
        self.load_config(self.config_path)

    def _start_config_watcher(self, config_path: str) -> None:
        """
        watchdogを使って設定ファイルの変更を監視するスレッドを起動する。

        Args:
            config_path (str): 監視対象の設定ファイルパス

        Raises:
            Exception: 監視起動失敗時に例外を送出する
        """
        try:
            self.observer = Observer()
            event_handler = ConfigChangeHandler(self)
            self.observer.schedule(event_handler, path=os.path.dirname(config_path), recursive=False)
            self.observer.start()
            self.logger.info(f"Watching configuration changes in {config_path}")
        except Exception:
            self.logger.exception("Failed to start config watcher.")
            raise

    def add_hook(self, hook_type: HookType, func: Callable[..., None], priority: int = 0) -> None:
        """
        指定のフックタイプに関数を優先度付きで登録する。

        Args:
            hook_type (HookType): フックの種類
            func (Callable[..., None]): 呼び出す関数
            priority (int): 優先度（大きいほど先に呼ばれる）
        """
        hook_list = self.hooks[hook_type]
        hook_list.append((priority, func))
        hook_list.sort(reverse=True, key=lambda x: x[0])

    def _execute_hooks(self, hook_type: HookType) -> None:
        """
        指定されたフックタイプの全登録関数を並列実行し、例外をロギングする。

        Args:
            hook_type (HookType): 実行するフックタイプ

        Raises:
            RuntimeError: フック実行中に例外があればhandle_errorで例外を送出する可能性あり
        """
        hooks = self.hooks[hook_type]
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(hook[1]) for hook in hooks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.exception(f"Error during hook execution: {e}")
                    errors.append(Exception("Hook execution failed."))

        if errors:
            self.handle_error(f"Errors encountered during {hook_type.value} hook execution: {errors}")

    def handle_error(self, message: str, raise_exception: bool = False) -> None:
        """
        エラーメッセージをログに出力し、必要なら例外を送出する。

        Args:
            message (str): ログに出すエラーメッセージ
            raise_exception (bool): TrueならRuntimeErrorを送出する
        """
        self.logger.exception(message)
        if raise_exception:
            raise RuntimeError(message)

    def _handle_shutdown(self, signum, frame):
        """
        シグナル受信時の終了処理。監視スレッド停止とcleanupを呼び出す。

        Args:
            signum: 受信したシグナル番号
            frame: 現在のスタックフレーム
        """
        self.logger.info(f"Received shutdown signal {signum}.")
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
        self.cleanup()

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
        """
        フェーズごとに、事前フック、メイン関数、事後フックを順に実行する。

        Args:
            pre_hook_type (HookType): 事前フックの種類
            post_hook_type (HookType): 事後フックの種類
            phase_function (Callable[[], None]): フェーズ本体の関数
            errors (List[Exception]): 発生した例外を格納するリスト
        """
        phase_name = pre_hook_type.name.split('_')[1].lower()
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
            self._execute_hooks(hook_type)
        except Exception:
            self.logger.exception(f"Error in {hook_type.name} hooks.")
            errors.append(Exception(f"{hook_type.name} hook error"))

    def _run_phase_function(self, func: Callable[[], None], name: str, errors: List[Exception]) -> None:
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
            self.logger.info(f"{name.capitalize()} phase completed in {duration:.2f} seconds.")
        except Exception as e:
            self.logger.exception(f"Error during {name} phase: {e}")
            errors.append(Exception(f"{name} phase error"))

    def setup(self) -> None:
        """
        バッチ処理のセットアップフェーズ。
        必要に応じて継承先でオーバーライドして初期化処理を実装する。
        """
        self.logger.info("Executing common setup tasks in BaseBatchProcessor.")

    def process(self) -> None:
        """
        バッチ処理のメイン処理フェーズ。
        get_dataで取得したデータをバッチ分割し、_process_batchで順次処理する。
        """
        self.logger.info("Executing common batch processing tasks in BaseBatchProcessor.")
        data = self.get_data()
        batches = self._generate_batches(data)
        for i, batch in enumerate(batches):
            self.logger.info(f"Processing batch {i + 1}...")
            self._process_batch(batch)

    @abstractmethod
    def get_data(self) -> List[Dict]:
        """
        処理対象データを取得する抽象メソッド。

        Returns:
            List[Dict]: 処理対象のデータリスト
        """
        pass

    @abstractmethod
    def _process_batch(self, batch: List[Dict]) -> None:
        """
        1バッチ分のデータを処理する抽象メソッド。

        Args:
            batch (List[Dict]): 処理対象のバッチデータ
        """
        pass

    def _generate_batches(self, data: List[Dict]) -> List[List[Dict]]:
        """
        処理データを設定値に従いバッチ分割する。

        Args:
            data (List[Dict]): 処理対象の全データ

        Returns:
            List[List[Dict]]: バッチ分割後のデータリスト
        """
        batch_size = self.config.get("batch_size", 100)
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def cleanup(self) -> None:
        """
        バッチ処理のクリーンアップフェーズ。
        後処理やリソース解放処理を継承先で実装する場合にオーバーライドする。
        """
        self.logger.info("Executing common cleanup tasks in BaseBatchProcessor.")
