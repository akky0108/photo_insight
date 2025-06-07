from batch_framework.core.hook_manager import HookManager, HookType


def test_add_and_execute_hooks_in_priority_order():
    results = []

    def hook_a():
        results.append("A")

    def hook_b():
        results.append("B")

    manager = HookManager(max_workers=1)
    manager.add_hook(HookType.PRE_SETUP, hook_b, priority=1)
    manager.add_hook(HookType.PRE_SETUP, hook_a, priority=2)

    manager.execute_hooks(HookType.PRE_SETUP)

    # priorityが高いhook_aが先に実行される
    assert results == ["A", "B"]


def test_execute_hooks_returns_no_errors():
    def hook():
        pass

    manager = HookManager()
    manager.add_hook(HookType.POST_PROCESS, hook)

    errors = manager.execute_hooks(HookType.POST_PROCESS)
    assert errors == []


def test_execute_hooks_with_exception_collects_errors():
    def good_hook():
        pass

    def bad_hook():
        raise RuntimeError("Test hook error")

    manager = HookManager()
    manager.add_hook(HookType.PRE_CLEANUP, good_hook)
    manager.add_hook(HookType.PRE_CLEANUP, bad_hook)

    errors = manager.execute_hooks(HookType.PRE_CLEANUP)

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "Test hook error" in str(errors[0])
