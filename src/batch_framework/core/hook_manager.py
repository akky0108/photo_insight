from enum import Enum
from typing import Callable, Dict, List, Tuple, NamedTuple
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


class HookManager:
    def __init__(self, max_workers: int = 2, logger=None):
        self.hooks: Dict[HookType, List[Hook]] = {
            hook_type: [] for hook_type in HookType
        }
        self.max_workers = max_workers
        self.logger = logger

    def add_hook(
        self, hook_type: HookType, func: Callable[..., None], priority: int = 0, parallel: bool = True
    ):
        self.hooks[hook_type].append(Hook(priority, func, parallel))
        self.hooks[hook_type].sort(reverse=True, key=lambda h: h.priority)

    def execute_hooks(self, hook_type: HookType) -> List[Exception]:
        errors = []
        hooks = self.hooks.get(hook_type, [])

        # 並列 / 直列 に振り分け
        serial_hooks = [hook for hook in hooks if not hook.parallel]
        parallel_hooks = [hook for hook in hooks if hook.parallel]

        # --- 直列実行 ---
        for hook in serial_hooks:
            try:
                hook.func()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"[Serial] Error during hook execution: {e}")
                errors.append(e)

        # --- 並列実行 ---
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(hook.func) for hook in parallel_hooks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[Parallel] Error during hook execution: {e}")
                    errors.append(e)

        return errors

