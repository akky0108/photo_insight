import time

from photo_insight.batch_framework.core.hook_manager import HookManager, HookType


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


def test_serial_hooks_run_in_order():
    results = []

    def hook1():
        results.append("1")

    def hook2():
        results.append("2")

    def hook3():
        results.append("3")

    manager = HookManager()
    manager.add_hook(HookType.POST_SETUP, hook3, priority=1, parallel=False)
    manager.add_hook(HookType.POST_SETUP, hook1, priority=3, parallel=False)
    manager.add_hook(HookType.POST_SETUP, hook2, priority=2, parallel=False)

    manager.execute_hooks(HookType.POST_SETUP)

    # priority順に直列で実行されているか確認
    assert results == ["1", "2", "3"]


def test_parallel_hooks_all_execute():
    results = []

    def make_hook(name):
        def hook():
            time.sleep(0.05)  # 並列確認用に少し待つ
            results.append(name)

        return hook

    manager = HookManager(max_workers=3)
    manager.add_hook(HookType.POST_CLEANUP, make_hook("A"), parallel=True)
    manager.add_hook(HookType.POST_CLEANUP, make_hook("B"), parallel=True)
    manager.add_hook(HookType.POST_CLEANUP, make_hook("C"), parallel=True)

    manager.execute_hooks(HookType.POST_CLEANUP)

    # 実行順は問わないがすべて呼ばれている
    assert sorted(results) == ["A", "B", "C"]


def test_mixed_parallel_and_serial_hooks():
    results = []

    def serial_hook():
        results.append("serial")

    def parallel_hook():
        results.append("parallel")

    manager = HookManager(max_workers=1)
    manager.add_hook(HookType.POST_SETUP, serial_hook, priority=1, parallel=False)
    manager.add_hook(HookType.POST_SETUP, parallel_hook, priority=1, parallel=True)

    manager.execute_hooks(HookType.POST_SETUP)

    assert "serial" in results
    assert "parallel" in results
