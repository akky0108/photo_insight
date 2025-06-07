from enum import Enum
from typing import Callable, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor


class HookType(Enum):
    PRE_SETUP = "pre_setup"
    POST_SETUP = "post_setup"
    PRE_PROCESS = "pre_process"
    POST_PROCESS = "post_process"
    PRE_CLEANUP = "pre_cleanup"
    POST_CLEANUP = "post_cleanup"


class HookManager:
    def __init__(self, max_workers: int = 2, logger=None):
        self.hooks: Dict[HookType, List[Tuple[int, Callable[[], None]]]] = {
            hook_type: [] for hook_type in HookType
        }
        self.max_workers = max_workers
        self.logger = logger

    def add_hook(
        self, hook_type: HookType, func: Callable[..., None], priority: int = 0
    ):
        self.hooks[hook_type].append((priority, func))
        self.hooks[hook_type].sort(reverse=True, key=lambda x: x[0])

    def execute_hooks(self, hook_type: HookType) -> List[Exception]:
        errors = []
        hooks = self.hooks.get(hook_type, [])
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(hook[1]) for hook in hooks]
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error during hook execution: {e}")
                    errors.append(e)
        return errors
