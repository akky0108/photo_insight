# src/batch_framework/base_batch.py
from __future__ import annotations

import os
import time
import logging
import inspect
from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Dict, Any

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- Optional deps (avoid ImportError crash at import time) ---
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs) -> None:  # type: ignore
        return

try:
    from watchdog.events import FileSystemEventHandler  # type: ignore
except Exception:  # pragma: no cover
    class FileSystemEventHandler:  # minimal fallback
        pass

from batch_framework.core.hook_manager import HookManager, HookType
from batch_framework.core.config_manager import ConfigManager
from batch_framework.core.signal_handler import SignalHandler
from batch_framework.utils.result_store import ResultStore, RunContext


def _normpath(p: str) -> str:
    """Path normalize for reliable comparisons."""
    return os.path.normcase(os.path.abspath(os.path.normpath(p)))


class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, processor: "BaseBatchProcessor"):
        """
        コンフィグファイルの変更を監視し、変更があればプロセッサに通知するハンドラクラス
        """
        self.processor = processor
        self._last_modified_time = 0.0

        # 監視対象パスを正規化して保持（event側の絶対パスと比較するため）
        self._target_config_path = _normpath(self.processor.config_path)

    def on_modified(self, event):
        """
        ファイルが変更された際に呼ばれるメソッド。短時間で連続変更された場合は無視する。
        """
        now = time.time()
        if now - self._last_modified_time < 1.0:
            return
        self._last_modified_time = now

        src_path = getattr(event, "src_path", "")
        if not src_path:
            return

        if _normpath(src_path) == self._target_config_path:
            self.processor.logger.info(
                f"Config file {src_path} has been modified. Reloading..."
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
        load_dotenv()

        self.project_root = os.getenv("PROJECT_ROOT") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.max_workers = max(1, int(max_workers or 0))

        resolved_config_path = (
            config_path
            if config_path is not None
            else os.path.join(self.project_root, "config", "config.yaml")
        )
        self.config_path = resolved_config_path

        # logger / config_manager wiring
        if logger is None:
            if config_manager is None:
                config_manager = ConfigManager(config_path=self.config_path)
            logger = config_manager.get_logger(self.__class__.__name__)
        else:
            if config_manager is None:
                config_manager = ConfigManager(config_path=self.config_path)

        self.logger = logger
        self.config_manager = config_manager

        if hook_manager is None:
            hook_manager = HookManager(max_workers=self.max_workers, logger=self.logger)
        else:
            hook_manager.logger = self.logger
        self.hook_manager = hook_manager

        if signal_handler is None:
            signal_handler = SignalHandler(self)
        signal_handler.logger = self.logger
        self.signal_handler = signal_handler

        self.processed_count = 0
        self.config = self.config_manager.config

        self.fail_fast = bool(self.config.get("batch", {}).get("fail_fast", True))
        self.cleanup_fail_fast = bool(self.config.get("batch", {}).get("cleanup_fail_fast", False))

        self._lock = Lock()

        # data cache (single-load contract)
        self._data_cache: Optional[List[Dict[str, Any]]] = None
        self._data_loaded: bool = False

        self._warn_if_get_data_overridden()

        # ============================================================
        # Result persistence (base)
        #
        # おすすめ方針:
        # - デフォルトは OFF（明示ON）
        #   → テストや通常実行で runs/ が勝手に作られない
        # - 保存したい場合だけ config で有効化する
        #
        # config例:
        # debug:
        #   persist_run_results: true
        #   run_results_dir: runs
        # ============================================================
        self._persist_run_results = bool(
            self.config.get("debug", {}).get("persist_run_results", False)
        )
        base_dir = self.config.get("debug", {}).get("run_results_dir", "runs")

        self.result_store = ResultStore(
            base_dir=base_dir,
            use_date_partition=True,
            final_dir=None,
        )
        self.run_ctx: Optional[RunContext] = None

        # optional: 同一インスタンスで複数 execute を呼ぶケース向け
        self.completed_all_batches = False
        self.final_summary: Dict[str, Any] = {}
        self.all_results: List[Dict[str, Any]] = []

    def execute(self, *args, **kwargs) -> None:
        self.start_time = time.time()

        # ===== ResultStore: start run (LAZY) =====
        # ここでは絶対にディレクトリを作らない（副作用ゼロ）
        # 実際に save_* が呼ばれたタイミングで atomic write が必要な親ディレクトリを作る
        try:
            if self._persist_run_results:
                self.run_ctx = self.result_store.make_run_context(
                    prefix=self.__class__.__name__.lower(),
                    ensure_dirs=False,  # ★重要: ここで作らない
                )
                self.logger.info(
                    f"[{self.__class__.__name__}] RunContext prepared: {self.run_ctx.out_dir}"
                )
        except Exception as e:
            self.logger.error(f"RunContext init failed: {e}", exc_info=True)
            self.run_ctx = None
        # ==================================

        self.logger.info(f"[{self.__class__.__name__}] Batch process started.")
        errors: List[Exception] = []

        try:
            self._execute_phase(
                HookType.PRE_SETUP, HookType.POST_SETUP, self.setup, errors
            )
            self._execute_phase(
                HookType.PRE_PROCESS,
                HookType.POST_PROCESS,
                lambda: self.process(*args, **kwargs),
                errors,
            )
        finally:
            self._execute_phase(
                HookType.PRE_CLEANUP, HookType.POST_CLEANUP, self.cleanup, errors
            )
            duration = time.time() - self.start_time
            self.logger.info(
                f"[{self.__class__.__name__}] Batch process completed in {duration:.2f} seconds."
            )

            if errors:
                msgs = [f"{type(e).__name__}: {e}" for e in errors]
                self.handle_error(
                    f"Batch process encountered errors: {msgs}", raise_exception=True
                )

            # ===== ResultStore: finalize =====
            if self.run_ctx and self._persist_run_results:
                try:
                    dst = self.result_store.finalize_to_final_dir(self.run_ctx)
                    if dst:
                        self.logger.info(f"Run results finalized to: {dst}")
                except Exception as e:
                    self.logger.error(f"finalize failed: {e}", exc_info=True)
            # ================================

    def _execute_phase(
        self,
        pre_hook_type: HookType,
        post_hook_type: HookType,
        phase_function: Callable[[], None],
        errors: List[Exception],
    ) -> None:
        phase_name = pre_hook_type.name.split("_")[1].lower()
        self._run_phase_hooks(pre_hook_type, errors)
        self._run_phase_function(phase_function, phase_name, errors)
        self._run_phase_hooks(post_hook_type, errors)

    def _run_phase_hooks(self, hook_type: HookType, errors: List[Exception]) -> None:
        try:
            self.hook_manager.execute_hooks(hook_type)
        except Exception as e:
            self.logger.error(
                f"[{self.__class__.__name__}] Error in {hook_type.name} hooks: {e}",
                exc_info=True,
            )
            if self.fail_fast:
                raise
            errors.append(e)

    def _run_phase_function(self, func: Callable[[], None], name: str, errors: List[Exception]) -> None:
        try:
            start_time = time.time()
            self.logger.info(f"[{self.__class__.__name__}] Executing {name} phase.")
            func()
            duration = time.time() - start_time
            self.logger.info(
                f"[{self.__class__.__name__}] {name.capitalize()} phase completed in {duration:.2f} seconds."
            )
        except Exception as e:
            self.logger.error(
                f"[{self.__class__.__name__}] Error during {name} phase: {e}",
                exc_info=True,
            )
            if self.fail_fast:
                raise
            errors.append(e)

    def add_hook(
        self, hook_type: HookType, func: Callable[[], None], priority: int = 0
    ) -> None:
        self.hook_manager.add_hook(hook_type, func, priority)

    def reload_config(self, config_path: Optional[str] = None) -> None:
        self.config_manager.reload_config(config_path)
        self.config = self.config_manager.config

        # config reload で persist_run_results を切り替えたい場合の追従
        self._persist_run_results = bool(
            self.config.get("debug", {}).get("persist_run_results", False)
        )
        # run_results_dir 変更はインスタンス差し替えが安全（運用で必要なら対応）
        # ここでは副作用を避けるため自動で作り直さない

    def handle_error(self, message: str, raise_exception: bool = False) -> None:
        self.logger.error(message)
        if raise_exception:
            raise RuntimeError(message)

    def _warn_if_get_data_overridden(self) -> None:
        """
        get_data() override は非推奨（Baseがキャッシュを握る契約を壊しやすい）。
        サブクラスは load_data() を実装してください。
        """
        try:
            if self.__class__.get_data is not BaseBatchProcessor.get_data:
                try:
                    src = inspect.getsourcefile(self.__class__)
                except Exception:
                    src = None
                src_note = f" ({src})" if src else ""
                self.logger.warning(
                    f"[{self.__class__.__name__}] Overriding get_data() is deprecated. "
                    f"Implement load_data() instead.{src_note}"
                )
        except Exception:
            return

    # =========================
    # Phases
    # =========================

    def setup(self) -> None:
        self.logger.info(f"[{self.__class__.__name__}] Executing common setup tasks.")
        self.data = self.get_data()
        self.after_data_loaded(self.data)

        # ===== ResultStore: save meta (only when enabled) =====
        if self.run_ctx and self._persist_run_results:
            try:
                self.result_store.save_meta(
                    self.run_ctx,
                    extra={
                        "processor": self.__class__.__name__,
                        "data_count": len(self.data),
                    },
                )
            except Exception as e:
                self.logger.error(f"save_meta failed: {e}", exc_info=True)
        # ==================================

    def after_data_loaded(self, data: List[Dict[str, Any]]) -> None:
        return

    def process(self, data: Optional[List[Dict[str, Any]]] = None) -> None:
        self.logger.info(
            f"[{self.__class__.__name__}] Executing common batch processing tasks."
        )

        if data is None:
            if hasattr(self, "data") and isinstance(self.data, list):
                data = self.data
            else:
                data = self.get_data()

        batches = self._generate_batches(data)
        failed_batches: List[int] = []
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

                future = executor.submit(self._safe_process_batch, batch)
                futures[future] = (i + 1, batch)

            for future in as_completed(futures):
                batch_idx, batch_data = futures[future]
                try:
                    batch_results = future.result()
                    if batch_results:
                        all_results.extend(batch_results)
                except Exception as e:
                    self.logger.error(
                        f"[{self.__class__.__name__}] [Batch {batch_idx}] Failed in thread: {e}",
                        exc_info=True,
                    )
                    truncated = str(batch_data)[:100]
                    self.logger.debug(
                        f"[{self.__class__.__name__}] [Batch {batch_idx}] Failed batch data (truncated): {truncated}"
                    )
                    failed_batches.append(batch_idx)

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
        self.all_results = all_results

        if failed_batches:
            self.logger.warning(
                f"[{self.__class__.__name__}] Processing completed with failures in batches: {failed_batches}"
            )
        else:
            self.logger.info(
                f"[{self.__class__.__name__}] All batches processed successfully."
            )
            self.completed_all_batches = True

        # ===== ResultStore: save results (only when enabled) =====
        if self.run_ctx and self._persist_run_results:
            try:
                self.result_store.save_jsonl(
                    self.run_ctx,
                    rows=all_results,
                    name="results.jsonl",
                )
            except Exception as e:
                self.logger.error(f"save_jsonl failed: {e}", exc_info=True)
        # ====================================

    def _safe_process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._process_batch(batch)

    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        success = [r for r in results if r.get("status") == "success"]
        failure = [r for r in results if r.get("status") != "success"]
        avg_score = (
            sum(self._safe_float_local(r.get("score", 0)) for r in success) / len(success)
            if success
            else None
        )
        return {
            "total": len(results),
            "success": len(success),
            "failure": len(failure),
            "avg_score": round(avg_score, 2) if avg_score is not None else None,
        }

    def _safe_float_local(self, v: Any) -> float:
        try:
            if v in ("", None):
                return 0.0
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    def cleanup(self) -> None:
        self.logger.info(f"[{self.__class__.__name__}] Executing common cleanup tasks.")

    def _generate_batches(
        self, data: List[Dict[str, Any]], batch_size: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        if not data:
            return []

        batch_size = (
            batch_size
            or getattr(self, "batch_size", None)
            or max(1, len(data) // self.max_workers)
        )
        return [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    def _should_stop_processing(self) -> bool:
        return getattr(self, "memory_threshold_exceeded", False)

    def _should_log_summary_detail(self) -> bool:
        return (
            os.getenv("DEBUG_LOG_SUMMARY") == "1"
            or self.config.get("debug", {}).get("log_summary_detail", False)
        )

    def get_lock(self) -> Lock:
        return self._lock

    # =========================
    # Data loading contract
    # =========================

    def get_data(self, *args, **kwargs) -> List[Dict[str, Any]]:
        if self._data_loaded and self._data_cache is not None:
            return self._data_cache

        try:
            data = self.load_data(*args, **kwargs)
        except NotImplementedError:
            data = self._legacy_get_data(*args, **kwargs)

        if data is None:
            data = []

        self._data_cache = data
        self._data_loaded = True
        return data

    @abstractmethod
    def load_data(self, *args, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _legacy_get_data(self, *args, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            "Please implement load_data() (preferred) or override get_data() (legacy)."
        )

    @abstractmethod
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError
