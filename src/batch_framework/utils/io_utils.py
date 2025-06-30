from collections import defaultdict
from pathlib import Path
from typing import List, Dict
import csv
import time
from threading import Lock


def group_by_key(data: List[Dict], key: str) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for item in data:
        group_value = item.get(key, "unknown")
        grouped[group_value].append(item)
    return grouped


def write_csv_with_lock(file_path: Path, data: List[Dict], fieldnames: List[str], lock: Lock, append: bool, logger) -> None:
    mode = "a" if append else "w"
    for attempt in range(3):
        try:
            is_new_file = not file_path.exists()
            with lock:
                with file_path.open(mode, newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    if mode == "w" or is_new_file:
                        writer.writeheader()
                    writer.writerows(data)
            logger.info(f"CSV出力成功: {file_path}")
            break
        except Exception as e:
            logger.error(f"CSV書き込み失敗 ({attempt + 1}回目): {e}")
            if attempt == 2:
                raise
            time.sleep(1)
