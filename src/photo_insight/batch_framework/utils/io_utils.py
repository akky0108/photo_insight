# src/batch_framework/utils/io_utils.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Iterable, Sequence, Union
import csv
import os
import time
from threading import Lock


def group_by_key(data: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for item in data:
        group_value = item.get(key, "unknown")
        grouped[group_value].append(item)
    return dict(grouped)


def _file_needs_header(path: Path) -> bool:
    """存在しない or 0 bytes ならヘッダが必要とみなす。"""
    try:
        return (not path.exists()) or path.stat().st_size == 0
    except FileNotFoundError:
        return True


def _iter_extra_keys(rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> List[str]:
    """fieldnames 以外に含まれているキー（参考ログ用）。"""
    fset = set(fieldnames)
    extras = set()
    for r in rows:
        for k in r.keys():
            if k not in fset:
                extras.add(k)
    return sorted(extras)


class _NoopLogger:
    def info(self, *args, **kwargs): ...
    def warning(self, *args, **kwargs): ...
    def error(self, *args, **kwargs): ...


def _ensure_logger(logger):
    return logger if logger is not None else _NoopLogger()


def _ensure_lock(lock):
    return lock if isinstance(lock, Lock.__class__) or hasattr(lock, "__enter__") else Lock()
    # ↑ Lock() は C実装で型判定が微妙なため「context managerならOK」も許容


def _infer_fieldnames(data: List[Dict[str, Any]]) -> List[str]:
    # 順序をなるべく安定させる：先頭行のキー順 + 後続で増えたキーを末尾に
    if not data:
        return []
    seen = set()
    out: List[str] = []
    for r in data:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def write_csv_with_lock(
    file_path: Union[str, Path],
    data: List[Dict[str, Any]],
    fieldnames: Sequence[str],
    lock,
    append: bool,
    logger,
    *,
    retries: int = 3,
    sleep_sec: float = 1.0,
    fsync: bool = True,
) -> None:
    """
    CSVをスレッドセーフに書く（同一プロセス内 Lock 前提）。

    - append=False: atomic write（tmpへ書いてreplace）
    - append=True : 追記（ヘッダ要否を厳密化）
    - extrasaction="ignore": 契約外キーは捨てる（列ズレ事故を防ぐ）
    - restval="": 欠損は空で埋める
    """
    logger = _ensure_logger(logger)
    lock = _ensure_lock(lock)
    file_path = Path(file_path)

    if not data:
        logger.info(f"CSV出力スキップ（データ0件）: {file_path}")
        return

    fn = list(fieldnames) if fieldnames is not None else []
    if not fn:
        fn = _infer_fieldnames(data)
        logger.warning(f"CSV出力: fieldnames が空のため推定しました: {fn} -> {file_path}")

    extra_keys = _iter_extra_keys(data, fn)
    if extra_keys:
        logger.warning(f"CSV出力: fieldnames外のキーを無視します: {extra_keys} -> {file_path}")

    for attempt in range(retries):
        try:
            with lock:
                file_path.parent.mkdir(parents=True, exist_ok=True)

                if append:
                    need_header = _file_needs_header(file_path)
                    with file_path.open("a", newline="", encoding="utf-8") as csvfile:
                        writer = csv.DictWriter(
                            csvfile,
                            fieldnames=fn,
                            extrasaction="ignore",
                            restval="",
                        )
                        if need_header:
                            writer.writeheader()
                        writer.writerows(data)

                        csvfile.flush()
                        if fsync:
                            os.fsync(csvfile.fileno())

                else:
                    tmp_path = file_path.with_suffix(
                        file_path.suffix + f".tmp.{os.getpid()}"
                    )
                    with tmp_path.open("w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.DictWriter(
                            csvfile,
                            fieldnames=fn,
                            extrasaction="ignore",
                            restval="",
                        )
                        writer.writeheader()
                        writer.writerows(data)

                        csvfile.flush()
                        if fsync:
                            os.fsync(csvfile.fileno())

                    # os.replace は同名が存在しても置換する（Windowsでも比較的安定）
                    os.replace(str(tmp_path), str(file_path))

            logger.info(f"CSV出力成功: {file_path}")
            return

        except Exception as e:
            logger.error(
                f"CSV書き込み失敗 ({attempt + 1}/{retries}): {file_path} err={e}",
                exc_info=True,
            )
            if attempt == retries - 1:
                raise
            time.sleep(sleep_sec)
