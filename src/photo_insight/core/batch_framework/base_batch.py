from __future__ import annotations

import inspect
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

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


from ._internal.config_manager import ConfigManager
from ._internal.hook_manager import HookManager, HookType
from ._internal.signal_handler import SignalHandler
from .utils.result_store import ResultStore, RunContext


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
        self._target_config_path = _normpath(self.processor.config_path) if self.processor.config_path else ""

    def on_modified(self, event) -> None:
        if not self._target_config_path:
            return

        now = time.time()
        if now - self._last_modified_time < 1.0:
            return
        self._last_modified_time = now

        src_path = getattr(event, "src_path", "")
        if not src_path:
            return

        if _normpath(src_path) == self._target_config_path:
            self.processor.logger.info(f"Config file {src_path} has been modified. Reloading...")
            self.processor.reload_config()


class BaseBatchProcessor(ABC):
    """
    Framework responsibility (minimal):
    - phase lifecycle + hook execution
    - batch submission / collection + summary
    - optional run persistence (ResultStore)
    - stop coordination (boolean + reason) as execution metadata
    - runtime parameter application before setup()

    Runtime parameter policy
    ------------------------
    - CLI / runner から execute(**kwargs) で渡された値は、
      processor 側が runtime_param_names に宣言したものだけを FW が属性反映する
    - process() には runtime kwargs を渡さない
    - processor は setup()/load_data()/_process_batch() で反映済み属性を参照する

    max_images policy
    -----------------
    - max_images は「実際に処理される件数の上限」とする
    - 判定は submit 前に行う
    - worker 内停止に依存しない
    - memory_threshold より max_images 到達を優先して submit を止める
    """

    #: 実行時に FW が属性へ反映してよい runtime parameter 名の一覧
    runtime_param_names: Tuple[str, ...] = ()

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_workers: int = 2,
        hook_manager: Optional[HookManager] = None,
        config_manager: Optional[ConfigManager] = None,
        signal_handler: Optional[SignalHandler] = None,
        logger: Optional[logging.Logger] = None,
        # ===== ConfigManager DI knobs =====
        config_env: Optional[str] = None,
        config_paths: Optional[List[str]] = None,
        resolver: Any = None,
        loader: Any = None,
        watch_factory: Any = None,
        list_policy: str = "replace",
        strict_missing: bool = True,
        auto_load: bool = True,
    ):
        load_dotenv()

        self.project_root = os.getenv("PROJECT_ROOT") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.max_workers = max(1, int(max_workers or 0))

        # resolved representative config path for watch/log
        self.config_path: str = ""

        # logger / config_manager wiring
        if config_manager is None:
            config_manager = ConfigManager(
                config_path=config_path,
                config_paths=config_paths,
                env=config_env,
                list_policy=list_policy,
                resolver=resolver,
                loader=loader,
                watch_factory=watch_factory,
                strict_missing=strict_missing,
                auto_load=auto_load,
            )

        self.config_path = getattr(config_manager, "config_path", "") or ""

        if logger is None:
            logger = config_manager.get_logger(self.__class__.__name__)

        self.logger = logger
        self.config_manager = config_manager

        if hook_manager is None:
            hook_manager = HookManager(max_workers=self.max_workers, logger=self.logger)
        else:
            hook_manager.logger = self.logger
        self.hook_manager = hook_manager

        # -------------------------
        # Stop coordination (base)
        # -------------------------
        self._stop_requested: bool = False
        self._stop_reason: Optional[str] = None  # e.g. memory_threshold / exception / signal_interrupt

        if signal_handler is None:
            signal_handler = SignalHandler(self._on_shutdown_signal, logger=self.logger)
        else:
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
        self.data: List[Dict[str, Any]] = []

        self._warn_if_get_data_overridden()

        # ============================================================
        # Result persistence (base)
        # ============================================================
        self._persist_run_results = bool(self.config.get("debug", {}).get("persist_run_results", False))
        base_dir = self.config.get("debug", {}).get("run_results_dir", "runs")

        self.result_store = ResultStore(
            base_dir=base_dir,
            use_date_partition=True,
            final_dir=None,
        )
        self.run_ctx: Optional[RunContext] = None

        # per-execute outputs
        self.completed_all_batches = False
        self.final_summary: Dict[str, Any] = {}
        self.all_results: List[Dict[str, Any]] = []
        self.start_time: float = 0.0

    # ============================================================
    # runtime params (base API)
    # ============================================================
    def get_runtime_param_names(self) -> Tuple[str, ...]:
        """
        FW が execute(**kwargs) から属性反映してよいパラメータ名一覧。
        必要に応じて subclass で runtime_param_names を宣言する。
        """
        return tuple(self.runtime_param_names or ())

    def apply_runtime_params(self, params: Dict[str, Any]) -> None:
        """
        execute(**kwargs) で渡された runtime parameter を setup() 前に属性へ反映する。
        宣言されていないキーは無視する。
        None は「未指定」とみなし反映しない。
        """
        allowed = set(self.get_runtime_param_names())
        if not allowed:
            if params:
                ignored = sorted(k for k, v in params.items() if v is not None)
                if ignored:
                    self.logger.debug(f"[{self.__class__.__name__}] Ignored runtime params: {ignored}")
            return

        applied: List[str] = []
        ignored: List[str] = []

        for name, value in params.items():
            if value is None:
                continue

            if name in allowed:
                setattr(self, name, value)
                applied.append(name)
            else:
                ignored.append(name)

        if applied:
            self.logger.debug(f"[{self.__class__.__name__}] Applied runtime params: {sorted(applied)}")
        if ignored:
            self.logger.debug(f"[{self.__class__.__name__}] Ignored runtime params: {sorted(ignored)}")

    def _reset_execution_state(self) -> None:
        """
        execute() ごとの揮発状態を初期化する。
        runtime parameter の変化に追従できるよう data cache もここで破棄する。
        """
        self._stop_requested = False
        self._stop_reason = None

        self._data_cache = None
        self._data_loaded = False
        self.data = []

        self.completed_all_batches = False
        self.final_summary = {}
        self.all_results = []

        self.run_ctx = None
        self.processed_count = 0

    # ============================================================
    # stop (base API)
    # ============================================================
    def request_stop(self, reason: str) -> None:
        """
        Framework-level stop request.
        - Subclasses may set their own flags; Base only stores metadata.
        """
        if not self._stop_requested:
            self._stop_requested = True
        if not self._stop_reason:
            self._stop_reason = reason

        # best-effort: if subclass has _stop_event (threading.Event), set it
        ev = getattr(self, "_stop_event", None)
        try:
            if ev is not None and hasattr(ev, "set"):
                ev.set()
        except Exception:
            pass

    def get_stop_reason(self) -> Optional[str]:
        if getattr(self, "memory_threshold_exceeded", False):
            return "memory_threshold"
        return self._stop_reason

    def _on_shutdown_signal(self) -> None:
        self.request_stop("signal_interrupt")

    # ============================================================
    # lifecycle
    # ============================================================
    def execute(self, *args, **kwargs) -> None:
        """
        Execute lifecycle:
        1. runtime params apply
        2. setup()
        3. process()
        4. cleanup()

        Notes
        -----
        - runtime kwargs は process() に渡さない
        - process() は共通ロジックを保ち、processor は属性を読む
        """
        if args:
            self.logger.debug(f"[{self.__class__.__name__}] execute() positional args are ignored by framework: {args}")

        self.start_time = time.time()

        # register signals once per process (idempotent)
        try:
            self.signal_handler.register()
        except Exception:
            self.logger.debug("Signal handler register failed", exc_info=True)

        # reset per-execute state
        self._reset_execution_state()

        # apply runtime parameters BEFORE setup()
        self.apply_runtime_params(kwargs)

        # ===== ResultStore: start run (LAZY) =====
        try:
            if self._persist_run_results:
                self.run_ctx = self.result_store.make_run_context(
                    prefix=self.__class__.__name__.lower(),
                    ensure_dirs=False,
                )
                self.logger.info(f"[{self.__class__.__name__}] RunContext prepared:{self.run_ctx.out_dir}")
        except Exception as e:
            self.logger.error(f"RunContext init failed: {e}", exc_info=True)
            self.run_ctx = None

        self.logger.info(f"[{self.__class__.__name__}] Batch process started.")
        errors: List[Exception] = []

        try:
            self._execute_phase(HookType.PRE_SETUP, HookType.POST_SETUP, self.setup, errors)
            self._execute_phase(
                HookType.PRE_PROCESS,
                HookType.POST_PROCESS,
                self.process,
                errors,
            )
        finally:
            self._execute_phase(HookType.PRE_CLEANUP, HookType.POST_CLEANUP, self.cleanup, errors)
            duration = time.time() - self.start_time
            self.logger.info(f"[{self.__class__.__name__}] Batch process completed in {duration:.2f} seconds.")

            if errors:
                if not self.get_stop_reason():
                    self.request_stop("exception")
                msgs = [f"{type(e).__name__}: {e}" for e in errors]
                self.handle_error(f"Batch process encountered errors: {msgs}", raise_exception=True)

            # ===== ResultStore: finalize =====
            if self.run_ctx and self._persist_run_results:
                try:
                    dst = self.result_store.finalize_to_final_dir(self.run_ctx)
                    if dst:
                        self.logger.info(f"Run results finalized to: {dst}")
                except Exception as e:
                    self.logger.error(f"finalize failed: {e}", exc_info=True)

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

    def add_hook(self, hook_type: HookType, func: Callable[[], None], priority: int = 0) -> None:
        self.hook_manager.add_hook(hook_type, func, priority)

    def reload_config(self, config_path: Optional[str] = None) -> None:
        self.config_manager.reload_config(config_path)
        self.config = self.config_manager.config
        try:
            self.config_path = getattr(self.config_manager, "config_path", "") or ""
        except Exception:
            pass

        self._persist_run_results = bool(self.config.get("debug", {}).get("persist_run_results", False))

    def handle_error(self, message: str, raise_exception: bool = False) -> None:
        self.logger.error(message)
        if raise_exception:
            raise RuntimeError(message)

    def _warn_if_get_data_overridden(self) -> None:
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

        if self.run_ctx and self._persist_run_results:
            try:
                self.result_store.save_meta(
                    self.run_ctx,
                    extra={
                        "processor": self.__class__.__name__,
                        "data_count": len(self.data),
                        "runtime_param_names": list(self.get_runtime_param_names()),
                    },
                )
            except Exception as e:
                self.logger.error(f"save_meta failed: {e}", exc_info=True)

    def after_data_loaded(self, data: List[Dict[str, Any]]) -> None:
        return

    def process(self, data: Optional[List[Dict[str, Any]]] = None) -> None:
        self.logger.info(f"[{self.__class__.__name__}] Executing common batch processing tasks.")

        # per-process reset
        self.completed_all_batches = False
        self.final_summary = {}
        self.all_results = []
        self.processed_count = 0

        if data is None:
            if isinstance(getattr(self, "data", None), list):
                data = self.data
            else:
                data = self.get_data()

        data = list(data or [])
        applied_max_images = self._resolve_max_images()
        total_input_items = len(data)

        batches = self._generate_batches(data)
        failed_batches: List[int] = []
        all_results: List[Dict[str, Any]] = []

        if not batches:
            self.logger.info("No data to process. Skipping batch execution.")
            self.final_summary = self._summarize_results(
                results=[],
                applied_max_images=applied_max_images,
                submitted_items=0,
                input_items=total_input_items,
            )
            return

        futures: Dict[Future, Tuple[int, List[Dict[str, Any]]]] = {}
        stop_requested = False
        submitted_batches = 0
        submitted_items = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit phase
            for i, batch in enumerate(batches):
                if not batch:
                    continue

                # max_images is evaluated BEFORE memory threshold.
                if applied_max_images is not None and submitted_items >= applied_max_images:
                    stop_requested = True
                    self.logger.info(
                        f"[{self.__class__.__name__}] Reached max_images before submitting batch {i + 1}: "
                        f"submitted_items={submitted_items}, max_images={applied_max_images}"
                    )
                    break

                if self._should_stop_processing():
                    stop_requested = True
                    self.logger.warning(
                        f"[{self.__class__.__name__}] Stopping before batch {i + 1} due to interrupt condition."
                    )
                    break

                batch_to_submit = batch
                if applied_max_images is not None:
                    remaining = applied_max_images - submitted_items
                    if remaining <= 0:
                        stop_requested = True
                        break
                    if len(batch_to_submit) > remaining:
                        batch_to_submit = batch_to_submit[:remaining]

                future = executor.submit(self._safe_process_batch, batch_to_submit)
                futures[future] = (i + 1, batch_to_submit)
                submitted_batches += 1
                submitted_items += len(batch_to_submit)

            # Collect phase
            cancelled_pending = False
            cancel_stop_reasons = {"memory_threshold", "signal_interrupt", "exception"}

            for future in as_completed(futures):
                batch_idx, batch_data = futures[future]

                current_stop_reason = self.get_stop_reason()
                should_cancel_pending = (
                    not cancelled_pending
                    and self._should_stop_processing()
                    and current_stop_reason in cancel_stop_reasons
                )

                if should_cancel_pending:
                    cancelled_pending = True
                    stop_requested = True
                    for f in futures:
                        if f is not future:
                            f.cancel()

                try:
                    batch_results = future.result()
                    if batch_results:
                        all_results.extend(batch_results)
                except CancelledError:
                    continue
                except Exception:
                    self.logger.exception(f"[{self.__class__.__name__}] [Batch {batch_idx}] Failed in thread")
                    truncated = str(batch_data)[:100]
                    self.logger.debug(
                        f"[{self.__class__.__name__}] [Batch {batch_idx}] Failed batch data "
                        f"(truncated): {truncated}"
                    )
                    failed_batches.append(batch_idx)
                    if self.fail_fast:
                        self.request_stop("exception")

            self.processed_count = len(all_results)

        if applied_max_images is not None and submitted_items >= applied_max_images and self.get_stop_reason() is None:
            self.request_stop("max_images_limit")

        summary = self._summarize_results(
            results=all_results,
            applied_max_images=applied_max_images,
            submitted_items=submitted_items,
            input_items=total_input_items,
        )
        self.logger.info(f"[{self.__class__.__name__}] Batch Summary: {summary}")

        self.logger.info(
            f"[{self.__class__.__name__}] Processed(actual) total={summary['total']} "
            f"success={summary['success']} failure={summary['failure']} "
            f"submitted_items={submitted_items} applied_max_images={applied_max_images}"
        )

        if self._should_log_summary_detail() and summary["avg_score"] is not None:
            self.logger.info(f"Average score: {summary['avg_score']}")

        self.final_summary = summary
        self.all_results = all_results

        interrupted = bool(stop_requested or self._should_stop_processing())

        if failed_batches and not self.get_stop_reason():
            self.request_stop("exception")

        if failed_batches:
            self.logger.warning(
                f"[{self.__class__.__name__}] Processing completed with failures in batches: {failed_batches}"
            )
        elif interrupted:
            self.logger.warning(
                f"[{self.__class__.__name__}] Processing stopped early. "
                f"submitted_batches={submitted_batches}/{len(batches)} "
                f"submitted_items={submitted_items}/{total_input_items} "
                f"stop_reason={self.get_stop_reason()}"
            )
        else:
            self.logger.info(f"[{self.__class__.__name__}] All batches processed successfully.")
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

            try:
                summary_out: Dict[str, Any] = dict(self.final_summary or {})
                summary_out.update(
                    {
                        "processor": self.__class__.__name__,
                        "data_count": len(data or []),
                        "input_items": int(total_input_items),
                        "processed_count": int(self.processed_count),
                        "max_workers": self.max_workers,
                        "config_path": self.config_path,
                        "runtime_param_names": list(self.get_runtime_param_names()),
                        "failed_batches": failed_batches,
                        "completed_all_batches": bool(self.completed_all_batches),
                        "interrupted": bool(interrupted),
                        "stop_reason": self.get_stop_reason(),
                        "submitted_batches": int(submitted_batches),
                        "total_batches": int(len(batches)),
                        "submitted_items": int(submitted_items),
                        "applied_max_images": applied_max_images,
                    }
                )

                if hasattr(self, "start_time"):
                    summary_out["duration_sec"] = round(time.time() - float(self.start_time), 3)

                self.result_store.save_json(
                    self.run_ctx,
                    obj=summary_out,
                    name="summary.json",
                )
            except Exception as e:
                self.logger.error(f"save_json failed: {e}", exc_info=True)

    def _safe_process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._process_batch(batch)

    def _resolve_max_images(self) -> Optional[int]:
        """
        max_images の適用値を解決する。

        Policy:
        - runtime param 等で self.max_images が設定されていればそれを優先
        - None / 0以下 / int化不可 は「未指定」とみなす
        """
        raw = getattr(self, "max_images", None)

        if raw in (None, ""):
            return None

        try:
            value = int(raw)
        except (TypeError, ValueError):
            self.logger.warning(f"[{self.__class__.__name__}] Invalid max_images={raw!r}; treated as unlimited.")
            return None

        if value <= 0:
            self.logger.warning(f"[{self.__class__.__name__}] Non-positive max_images={value}; treated as unlimited.")
            return None

        return value

    def _summarize_results(
        self,
        results: List[Dict[str, Any]],
        applied_max_images: Optional[int] = None,
        submitted_items: Optional[int] = None,
        input_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Summary policy:
        - success: status == "success"
        - avg_score: success のうち score が None/"" ではないものだけで平均（無いなら None）
        - total: 実際に返却された result 件数
        """
        success = [r for r in results if r.get("status") == "success"]
        failure = [r for r in results if r.get("status") != "success"]

        scores: List[float] = []
        for r in success:
            v = r.get("score", None)
            if v in ("", None):
                continue
            try:
                scores.append(float(v))
            except Exception:
                continue

        avg_score = (sum(scores) / len(scores)) if scores else None

        return {
            "total": len(results),
            "success": len(success),
            "failure": len(failure),
            "avg_score": round(avg_score, 2) if avg_score is not None else None,
            "applied_max_images": applied_max_images,
            "submitted_items": submitted_items,
            "input_items": input_items,
            "stop_reason": self.get_stop_reason(),
        }

    def cleanup(self) -> None:
        self.logger.info(f"[{self.__class__.__name__}] Executing common cleanup tasks.")

    def _generate_batches(
        self,
        data: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        if not data:
            return []

        batch_size = batch_size or getattr(self, "batch_size", None) or max(1, len(data) // self.max_workers)
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    def _should_stop_processing(self) -> bool:
        if getattr(self, "memory_threshold_exceeded", False):
            if not self._stop_reason:
                self._stop_reason = "memory_threshold"
            return True
        if self._stop_requested:
            return True
        return False

    def _should_log_summary_detail(self) -> bool:
        return os.getenv("DEBUG_LOG_SUMMARY") == "1" or self.config.get("debug", {}).get("log_summary_detail", False)

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
        raise NotImplementedError("Please implement load_data() (preferred) or override get_data() (legacy).")

    @abstractmethod
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        raise NotImplementedError
