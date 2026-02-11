# src/batch_framework/core/hook_manager.py
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional
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
        msg = f"{hook_type.name} hooks failed: " + ", ".join(
            f"{type(e).__name__}: {e}" for e in errors
        )
        super().__init__(msg)


class HookManager:
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
    ):
        self.hooks[hook_type].append(Hook(priority, func, parallel))
        self.hooks[hook_type].sort(reverse=True, key=lambda h: h.priority)

    def execute_hooks(self, hook_type: HookType) -> None:
        hooks = self.hooks.get(hook_type, [])
        if not hooks:
            return

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
                # fail-fast for hooks: break early
                break

        # --- parallel ---
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

        if errors:
            raise HookExecutionError(hook_type, errors)
