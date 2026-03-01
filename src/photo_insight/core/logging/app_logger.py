"""
utils.app_logger

プロジェクト全体で共通して使えるログユーティリティ。

- ロギング設定は YAML ファイルで読み込む
- Singleton パターンで logger インスタンスを管理
- cleanup 処理など便利機能付き
- 本番運用向けに logging config の探索優先順位を改善

探索優先順位:
1) 環境変数 PHOTO_INSIGHT_LOG_CONFIG
2) /work/config/logging_config.yaml
3) <project_root>/config/logging_config.yaml
4) <this_file_dir>/config/logging_config.yaml
"""

from __future__ import annotations

import atexit
import logging
import logging.config
import os
import threading
from pathlib import Path
from typing import Optional

import yaml


class AppLogger:
    """
    アプリケーション用のロガー管理クラス（Singleton）。

    引数:
        project_root (Optional[str]): プロジェクトルートパス（互換用）
        config_file (Optional[str]): 明示的にロギング設定ファイルを指定したい場合のパス
        logger_name (str): 作成するロガーの名前
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        project_root: Optional[str] = None,
        config_file: Optional[str] = None,
        logger_name: str = "MyAppLogger",
    ):
        # Singleton だが、logger_name は呼び出し側で変えたいことがある。
        # 初期化済みでも logger_name だけは反映する（最小改善）。
        if getattr(self, "_initialized", False):
            # 既存ロガー名と違えば差し替え（ハンドラは dictConfig/bas icConfig が持つので触らない）
            if getattr(self, "_logger_name", None) != logger_name:
                self.logger = logging.getLogger(logger_name)
                self._logger_name = logger_name
            return

        if project_root is None:
            project_root = os.getcwd()

        self.project_root = str(project_root)
        self._logger_name = logger_name

        # ログディレクトリ（互換: project_root/logs）
        try:
            log_dir = Path(self.project_root) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # ディレクトリが作れない環境でも落とさない
            pass

        # logging config の解決
        resolved = self._resolve_logging_config_path(project_root=self.project_root, config_file=config_file)

        # YAMLファイルからロギング設定を読み込む
        if resolved is not None and resolved.exists():
            try:
                with resolved.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                if not isinstance(cfg, dict):
                    raise ValueError("Invalid config format (expected dict)")
                logging.config.dictConfig(cfg)
                print(f"Logging configured from {resolved}.")
            except (yaml.YAMLError, ValueError, FileNotFoundError, OSError) as e:
                print(f"Error loading logging config: {e}. Using default logging settings.")
                logging.basicConfig(level=logging.INFO)
        else:
            # 候補を出す（運用で原因が分かる）
            searched = self._logging_config_candidates(project_root=self.project_root, config_file=config_file)
            print("Config file not found. Using default logging settings.")
            print("Searched candidates:")
            for p in searched:
                print(f"  - {p}")
            logging.basicConfig(level=logging.INFO)

        # ロガーを設定
        self.logger = logging.getLogger(logger_name)

        # プログラム終了時にクリーンアップを登録
        atexit.register(self.cleanup)

        self._initialized = True

    # -----------------------------
    # config path resolution
    # -----------------------------
    def _logging_config_candidates(self, project_root: str, config_file: Optional[str]) -> list[Path]:
        candidates: list[Path] = []

        # 0) explicit arg
        if config_file:
            candidates.append(Path(config_file))

        # 1) env var
        env = os.getenv("PHOTO_INSIGHT_LOG_CONFIG")
        if env:
            candidates.append(Path(env))

        # 2) production mount (your desired default)
        candidates.append(Path("/work/config/logging_config.yaml"))

        # 3) project_root/config (legacy)
        if project_root:
            candidates.append(Path(project_root) / "config" / "logging_config.yaml")

        # 4) relative to this file (last resort / legacy)
        try:
            here = Path(__file__).resolve().parent
            candidates.append(here / "config" / "logging_config.yaml")
        except Exception:
            pass

        # de-dup while keeping order
        out: list[Path] = []
        seen: set[str] = set()
        for p in candidates:
            key = str(p)
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    def _resolve_logging_config_path(self, project_root: str, config_file: Optional[str]) -> Optional[Path]:
        for p in self._logging_config_candidates(project_root=project_root, config_file=config_file):
            try:
                if p.exists():
                    return p
            except Exception:
                continue
        return None

    # -----------------------------
    # logger facade (optional)
    # -----------------------------
    def _log(self, level: int, message: str):
        if self.isEnabledFor(level):
            self.logger.log(level, message)

    def info(self, message: str):
        self._log(logging.INFO, message)

    def error(self, message: str):
        self._log(logging.ERROR, message)

    def debug(self, message: str):
        self._log(logging.DEBUG, message)

    def warning(self, message: str):
        self._log(logging.WARNING, message)

    def critical(self, message: str):
        self._log(logging.CRITICAL, message)

    def log_metric(self, metric_name: str, score: float) -> None:
        self.info(f"{metric_name.capitalize()} score: {score}")

    def change_logger_name(self, new_name: str):
        logging.Logger.manager.loggerDict.pop(self.logger.name, None)
        self.logger = logging.getLogger(new_name)
        self._logger_name = new_name

    def get_logger(self):
        return self.logger

    def cleanup(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            handler.close()

    def isEnabledFor(self, level: int) -> bool:
        return self.logger.isEnabledFor(level)


# エイリアスとして公開
Logger = AppLogger