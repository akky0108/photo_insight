# memory_monitor.py
import psutil
import os


class MemoryMonitor:
    def __init__(self, logger):
        self.logger = logger

    def get_memory_usage(self) -> float:
        """現在のメモリ使用量をパーセントで返す"""
        memory = psutil.virtual_memory()
        return memory.percent  # メモリ使用率（パーセント）を返す

    def log_usage(self, prefix: str = "") -> None:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)  # MB
        self.logger.info(f"[{prefix}] Memory usage: {mem:.2f} MB")
