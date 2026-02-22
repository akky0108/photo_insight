# src/batch_framework/core/hook_manager.py
from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, List, NamedTuple
from concurrent.futures import ThreadPoolExecutor


class Hook(NamedTuple):
    priority: int
    func: Callable[[], None]
    parallel: bool = True


class HookType(Enum):
    PRE_SETUP = "pre_setup"
    POST_SETUP = "post_setup"
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    PRE_CLEANUP = "pre_cleanup"
    POST_CLEANUP = "post_cleanup"


class HookExecutionError(RuntimeError):
    """Raised when one or more hooks fail."""

    def __init__(self, hook_type: HookType, errors: List[BaseException]):
        self.hook_type = hook_type
        self.errors = errors
        msg = f"{hook_type.name} hooks failed: " + ", ".join(f"{type(e).__name__}: {e}" for e in errors)
        super().__init__(msg)


class HookManager:
    """
    Hook execution policy:
    - execute_hooks() returns List[BaseException] (empty if ok)
    - caller can choose:
        - raise_on_error=True -> raise HookExecutionError if any error
        - fail_fast=True      -> stop serial hooks on first error
          (parallel still collects if reached)
    """

    def __init__(self, max_workers: int = 2, logger=None):
        self.hooks: Dict[HookType, List[Hook]] = {hook_type: [] for hook_type in HookType}
        self.max_workers = max(1, int(max_workers or 0))
        self.logger = logger

    def add_hook(
        self,
        hook_type: HookType,
        func: Callable[..., None],
        priority: int = 0,
        parallel: bool = True,
    ) -> None:
        # NOTE: func signature is kept loose, but execution uses func() (no args)
        self.hooks[hook_type].append(Hook(priority, func, parallel))
        self.hooks[hook_type].sort(reverse=True, key=lambda h: h.priority)

    def execute_hooks(
        self,
        hook_type: HookType,
        *,
        raise_on_error: bool = False,
        fail_fast: bool = False,
    ) -> List[BaseException]:
        """
        Execute hooks and return collected errors (never None).

        Args:
            raise_on_error: if True, raise HookExecutionError when errors exist.
            fail_fast: if True, stop executing *serial* hooks at first error.
                (parallel hooks are executed only if serial phase produced no errors)

        Returns:
            List of exceptions raised by hooks (empty list if all ok).
        """
        hooks = self.hooks.get(hook_type, [])
        if not hooks:
            return []

        serial_hooks = [hook for hook in hooks if not hook.parallel]
        parallel_hooks = [hook for hook in hooks if hook.parallel]

        errors: List[BaseException] = []

        # --- serial ---
        for hook in serial_hooks:
            try:
                hook.func()
            except BaseException as e:
                if self.logger:
                    self.logger.error(
                        f"[{hook_type.name}] [Serial] hook failed: {e}",
                        exc_info=True,
                    )
                errors.append(e)
                if fail_fast:
                    break

        # --- parallel ---
        # Keep existing behavior: don't run parallel hooks if serial already failed.
        if parallel_hooks and not errors:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(hook.func) for hook in parallel_hooks]
                for future in futures:
                    try:
                        future.result()
                    except BaseException as e:
                        if self.logger:
                            self.logger.error(
                                f"[{hook_type.name}] [Parallel] hook failed: {e}",
                                exc_info=True,
                            )
                        errors.append(e)

        if errors and raise_on_error:
            raise HookExecutionError(hook_type, errors)

        return errors

    # ---- compatibility helper (optional) ----
    def execute_hooks_or_raise(
        self,
        hook_type: HookType,
        *,
        fail_fast: bool = True,
    ) -> None:
        """
        Backward-compatible behavior: raise on any error.
        """
        self.execute_hooks(hook_type, raise_on_error=True, fail_fast=fail_fast)
