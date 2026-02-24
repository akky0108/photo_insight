"""
src/batch_framework/_internal/config_manager.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Union

import yaml
from dotenv import load_dotenv

# watchdog optional import (runtime)
try:
    from watchdog.events import FileSystemEventHandler
except Exception:  # pragma: no cover

    class FileSystemEventHandler:  # minimal fallback
        pass


# -------------------------
# Protocols (DI points)
# -------------------------
class ConfigSourceResolver(Protocol):
    def resolve_files(
        self,
        *,
        project_root: str,
        config_path: Optional[str],
        config_paths: Optional[List[str]],
        env: Optional[str],
        logger: Any,
    ) -> List[Path]: ...


class ConfigLoader(Protocol):
    def load(self, files: List[Path], *, logger: Any) -> Dict[str, Any]: ...


class WatchHandle(Protocol):
    def stop(self) -> None: ...


class WatchFactory(Protocol):
    def start(
        self,
        *,
        files: List[Path],
        on_change: Callable[[], None],
        logger: Any,
    ) -> WatchHandle: ...


# -------------------------
# helpers
# -------------------------
def _normpath(p: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(p))))


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: Any, override: Any, *, list_policy: str = "replace") -> Any:
    if override is None:
        return base

    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _deep_merge(out.get(k), v, list_policy=list_policy) if k in out else v
        return out

    if isinstance(base, list) and isinstance(override, list):
        if list_policy == "replace":
            return list(override)
        if list_policy == "concat":
            return list(base) + list(override)
        if list_policy == "concat_unique":
            seen = set()
            out: list[Any] = []
            for x in list(base) + list(override):
                key = repr(x)
                if key in seen:
                    continue
                seen.add(key)
                out.append(x)
            return out
        return list(override)

    return override


# -------------------------
# Default implementations
# -------------------------
class DefaultConfigLoader:
    def __init__(self, *, list_policy: str = "replace") -> None:
        self.list_policy = list_policy

    def load(self, files: List[Path], *, logger: Any) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for p in files:
            data = _read_yaml(p)
            data.pop("extends", None)
            merged = _deep_merge(merged, data, list_policy=self.list_policy)
        return merged


class DefaultConfigResolver:
    """
    いまの ConfigManager の resolve/extends ロジックを外に出しただけのもの。
    """

    def __init__(self, *, strict_missing: bool = True) -> None:
        self.strict_missing = strict_missing

    def resolve_files(
        self,
        *,
        project_root: str,
        config_path: Optional[str],
        config_paths: Optional[List[str]],
        env: Optional[str],
        logger: Any,
    ) -> List[Path]:
        cfg_dir = Path(project_root) / "config"
        default_single = cfg_dir / "config.yaml"

        # 1) explicit list
        if config_paths:
            return self._expand_extends([Path(p) for p in config_paths])

        # 2) single explicit path
        if config_path:
            return self._expand_extends([Path(config_path)])

        # 3) ENV / default
        env_final = (env or os.getenv("CONFIG_ENV", "")).strip()
        cfg_path = os.getenv("CONFIG_PATH", "").strip()
        cfg_base = os.getenv("CONFIG_BASE", "").strip()

        files: List[Path] = []
        if cfg_base:
            files.append(Path(cfg_base))
        if cfg_path:
            files.append(Path(cfg_path))
        if files:
            return self._expand_extends(files)

        if env_final:
            base = cfg_dir / "config.base.yaml"
            envf = cfg_dir / f"config.{env_final}.yaml"
            files = []
            if base.exists():
                files.append(base)
            files.append(envf)
            return self._expand_extends(files)

        return self._expand_extends([default_single])

    def _expand_extends(self, files: List[Path]) -> List[Path]:
        resolved: List[Path] = list(files)

        i = 0
        while i < len(resolved):
            p = resolved[i]
            if not p.exists():
                if self.strict_missing:
                    raise FileNotFoundError(f"Config file not found: {p}")
                # lenient: ないものは読み飛ばす（テスト向け）
                resolved.pop(i)
                continue

            data = _read_yaml(p)
            ext = data.get("extends")
            if not ext:
                i += 1
                continue

            if isinstance(ext, str):
                ext_list = [ext]
            elif isinstance(ext, list):
                ext_list = [str(x) for x in ext]
            else:
                raise ValueError(f"Invalid extends in {p}: {ext}")

            base_dir = p.parent
            insert_paths = []
            for x in ext_list:
                pp = Path(x)
                insert_paths.append(pp if pp.is_absolute() else (base_dir / pp))

            for pp in insert_paths[::-1]:
                if pp not in resolved:
                    resolved.insert(i, pp)
            i += 1

        return [Path(str(p)) for p in resolved]


class _NullWatch(WatchHandle):
    def stop(self) -> None:
        return


class NullWatchFactory:
    def start(self, *, files: List[Path], on_change: Callable[[], None], logger: Any) -> WatchHandle:
        return _NullWatch()


class ConfigChangeHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[], None], target_paths: List[Path]):
        self.callback = callback
        self._last_modified_time = 0.0
        self._targets = {_normpath(p) for p in target_paths}

    def on_modified(self, event):
        now = time.time()
        if now - self._last_modified_time < 1.0:
            return
        self._last_modified_time = now

        src_path = getattr(event, "src_path", "")
        if src_path and _normpath(src_path) in self._targets:
            self.callback()


class WatchdogFactory:
    """
    watchdog が無い / 起動失敗 → NullWatch にフォールバック。
    """

    def start(self, *, files: List[Path], on_change: Callable[[], None], logger: Any) -> WatchHandle:
        try:
            from watchdog.observers import Observer
        except Exception as e:
            if logger:
                logger.warning(f"Config watching disabled (watchdog not available): {e}")
            return _NullWatch()

        handler = ConfigChangeHandler(callback=on_change, target_paths=files)
        observer = Observer()
        dirs = sorted({str(p.parent) for p in files})
        for d in dirs:
            observer.schedule(handler, path=d, recursive=False)
        observer.start()

        class _Handle(WatchHandle):
            def stop(self_nonlocal) -> None:  # type: ignore[no-redef]
                observer.stop()
                observer.join()

        return _Handle()


# -------------------------
# ConfigManager (DI-ready)
# -------------------------
class ConfigManager:
    def __init__(
        self,
        config_path: Optional[str] = None,
        logger=None,
        *,
        config_paths: Optional[List[str]] = None,
        env: Optional[str] = None,
        list_policy: str = "replace",
        # DI points
        resolver: Optional[ConfigSourceResolver] = None,
        loader: Optional[ConfigLoader] = None,
        watch_factory: Optional[WatchFactory] = None,
        # policy
        strict_missing: bool = True,
        # test helper: そもそも読み込みを抑制（必要なら）
        auto_load: bool = True,
    ):
        load_dotenv()
        self.logger = logger
        self.project_root = os.getenv("PROJECT_ROOT") or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

        self._resolver = resolver or DefaultConfigResolver(strict_missing=strict_missing)
        self._loader = loader or DefaultConfigLoader(list_policy=list_policy)
        self._watch_factory = watch_factory or WatchdogFactory()

        self._config_files: List[Path] = self._resolver.resolve_files(
            project_root=self.project_root,
            config_path=config_path,
            config_paths=config_paths,
            env=env,
            logger=self.logger,
        )
        self.config_path = str(self._config_files[-1]) if self._config_files else ""

        self.config: Dict[str, Any] = {}
        self._watch: Optional[WatchHandle] = None

        if auto_load:
            self.load_config()

    def load_config(self) -> None:
        if self.logger:
            self.logger.info("Loading configuration from: " + " -> ".join(str(p) for p in self._config_files))
        merged = self._loader.load(self._config_files, logger=self.logger)
        self.config.clear()
        self.config.update(merged)

    def reload_config(self, config_path: Optional[str] = None) -> None:
        if config_path:
            # 代表パスだけ差し替え（従来互換）
            self._config_files = self._resolver.resolve_files(
                project_root=self.project_root,
                config_path=config_path,
                config_paths=None,
                env=None,
                logger=self.logger,
            )
            self.config_path = str(self._config_files[-1]) if self._config_files else ""

        if self.logger:
            self.logger.info("Reloading configuration.")
        self.load_config()

    def start_watching(self, on_change_callback: Callable[[], None]) -> None:
        # 既存watchを停止してから
        self.stop_watching()
        self._watch = self._watch_factory.start(
            files=self._config_files,
            on_change=on_change_callback,
            logger=self.logger,
        )

    def stop_watching(self) -> None:
        if self._watch:
            self._watch.stop()
            self._watch = None

    # getters (そのまま)
    def get_config(self) -> dict:
        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return (self.config or {}).get(key, default)
        except Exception:
            return default

    def get_path(
        self,
        path: Union[str, Iterable[str]],
        default: Any = None,
        *,
        sep: str = ".",
    ) -> Any:
        try:
            keys = path.split(sep) if isinstance(path, str) else list(path)
            cur: Any = self.config or {}
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur
        except Exception:
            return default

    def get_memory_threshold(self, default: int = 90) -> int:
        value = self.config.get("batch", {}).get("memory_threshold", default)
        try:
            value = int(value)
            if 1 <= value <= 100:
                return value
            if self.logger:
                self.logger.warning(f"Invalid memory_threshold: {value}. " f"Using default: {default}")
        except (ValueError, TypeError):
            if self.logger:
                self.logger.warning(f"Invalid memory_threshold format: {value}. " f"Using default: {default}")
        return default

    def get_logger(self, logger_name: Optional[str] = None):
        """
        設定に基づいて logger を生成する。

        優先順位:
        1) utils.app_logger.Logger が利用可能ならそれを使用
        2) 無ければ標準 logging にフォールバック

        設定例（任意）:
        logging:
          level: INFO
        """
        import logging

        if not logger_name:
            logger_name = self.__class__.__name__

        # -----------------------------
        # 1) try project Logger
        # -----------------------------
        try:
            from photo_insight.utils.app_logger import Logger as AppLogger  # type: ignore  # noqa: E501,E402

            return AppLogger(
                project_root=self.project_root,
                logger_name=logger_name,
            ).get_logger()

        except Exception:
            # utils.app_logger が無い環境（テスト等）では標準loggingへフォールバック
            pass

        # -----------------------------
        # 2) fallback: standard logging
        # -----------------------------
        logger = logging.getLogger(logger_name)

        # 既にハンドラが設定済みならそのまま返す（二重設定防止）
        if logger.handlers:
            return logger

        level_str = (self.config.get("logging", {}) or {}).get("level", "INFO")
        try:
            level = getattr(logging, str(level_str).upper())
        except Exception:
            level = logging.INFO

        logger.setLevel(level)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

        return logger
