import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
from threading import Lock
from collections import defaultdict

from file_handler.exif_file_handler import ExifFileHandler
from batch_framework.base_batch import BaseBatchProcessor
from batch_framework.utils.io_utils import group_by_key, write_csv_with_lock

ExifData = Dict[str, str]


class NEFFileBatchProcess(BaseBatchProcessor):
    """RAW (NEF) ファイルのバッチ処理クラス"""

    def __init__(self, config_path=None, max_workers=4):
        super().__init__(config_path=config_path, max_workers=max_workers)

        self._cached_data: Optional[List[Dict[str, Any]]] = None
        self._cache_key: Optional[str] = None

        self.exif_handler = ExifFileHandler()

        self.exif_fields = self.config.get(
            "exif_fields",
            [
                "FileName",
                "Model",
                "Lens",
                "ISO",
                "Aperture",
                "FocalLength",
                "Rating",
                "ImageHeight",
                "ImageWidth",
                "Orientation",
                "BitDepth",
            ],
        )

        self.append_mode = self.config.get("append_mode", False)

        self.base_directory_path = Path(
            self.config.get("base_directory_root", "/mnt/l/picture/2025")
        )

        self.output_directory = self.config.get("output_directory", "temp")

        self._csv_locks: Dict[str, Lock] = defaultdict(Lock)

        # 集計用
        self.output_data: List[Dict] = []
        self.success_count: int = 0
        self.failure_count: int = 0

    # ------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------
    def setup(self) -> None:
        self._cached_data = None
        self._cache_key = None

        # ★ 必須：常に定義
        self.target_dirs: List[Path] = []

        self.output_data.clear()
        self.success_count = 0
        self.failure_count = 0

        self.temp_dir = Path(self.project_root) / self.output_directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        base_dir = self.base_directory_path
        if not base_dir.exists():
            self.handle_error(f"ディレクトリが見つかりません: {base_dir}", raise_exception=True)

        self.target_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        self.logger.info(f"初期設定完了: 画像ディレクトリ {base_dir}")

        """初期化処理（data はここで作らない）"""
        #super().setup()

    def cleanup(self) -> None:
        super().cleanup()
        self.logger.info("クリーンアップ完了")

    # ------------------------------------------------------------
    # data collection
    # ------------------------------------------------------------
    def get_data(self, target_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        # ★ ① ここで「実際の target_dir」を確定させる
        if target_dir is None:
            target_dir = getattr(self, "target_dir", None)

        cache_key = self._make_cache_key(target_dir)

        # 🔑 キャッシュヒット
        if self._cached_data is not None and self._cache_key == cache_key:
            self.logger.info(
                f"get_data(): キャッシュ使用 ({len(self._cached_data)} 件)"
            )
            return self._cached_data

        # 🔄 キャッシュミス（キー違い or 初回）
        if self._cache_key != cache_key:
            self.logger.info(
                f"get_data(): キャッシュ更新 "
                f"(old={self._cache_key}, new={cache_key})"
            )

        self._cache_key = cache_key
        self._cached_data = None

        nef_files: List[Path] = []

        if target_dir:
            self.logger.info(f"指定ディレクトリのみ処理: {target_dir}")
            nef_files = list(target_dir.rglob("*.NEF"))
            self.logger.info(f"{target_dir} から {len(nef_files)} 件検出")
        else:
            self.logger.info("全ディレクトリを対象に処理")
            for d in self.target_dirs:
                found = list(d.rglob("*.NEF"))
                self.logger.info(f"{d} から {len(found)} 件検出")
                nef_files.extend(found)

        self._cached_data = [
            {
                "path": path,
                "subdir_name": path.parent.name,
            }
            for path in nef_files
        ]

        self.logger.info(f"get_data(): 収集ファイル数 = {len(self._cached_data)}")
        return self._cached_data

    # ------------------------------------------------------------
    # batch processing
    # ------------------------------------------------------------
    def _generate_batches(self, data: List[Dict]) -> List[List[Dict]]:
        grouped = group_by_key(data, "subdir_name")
        return list(grouped.values())

    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        if not batch:
            return []

        grouped = group_by_key(batch, "subdir_name")
        all_exif_data = []

        for subdir_name, items in grouped.items():
            exif_raw_list = []

            output_file_path = (
                self.temp_dir / f"{subdir_name}_raw_exif_data.csv"
            )

            # ★ 永続キャッシュ：既に出力済みならスキップ
            if output_file_path.exists() and not self.append_mode:
                self.logger.info(
                    f"skip (already exists): {output_file_path}"
                )
                continue

            for item in items:
                exif_raw = self.exif_handler.read_file(str(item["path"]))
                if exif_raw:
                    exif_raw_list.append(exif_raw)

            if not exif_raw_list:
                continue

            exif_data_list = self.filter_exif_data(exif_raw_list)

            output_file_path = (
                self.temp_dir / f"{subdir_name}_raw_exif_data.csv"
            )

            write_csv_with_lock(
                file_path=output_file_path,
                data=exif_data_list,
                fieldnames=self.exif_fields,
                lock=self._get_lock_for_file(output_file_path),
                append=self.append_mode,
                logger=self.logger,
            )

            all_exif_data.extend(exif_data_list)

        with self.get_lock():
            self.output_data.extend(all_exif_data)
            self.success_count += len(all_exif_data)

        return [{"status": "success"} for _ in all_exif_data]

    # ------------------------------------------------------------
    # utils
    # ------------------------------------------------------------
    def filter_exif_data(self, raw_files: List[ExifData]) -> List[ExifData]:
        results = []
        for raw in raw_files:
            filtered = {
                field: raw.get(field, "N/A") for field in self.exif_fields
            }
            results.append(filtered)
        return results

    def _get_lock_for_file(self, file_path: Path) -> Lock:
        key = str(file_path.resolve())
        return self._csv_locks[key]

    def _make_cache_key(self, target_dir: Optional[Path]) -> str:
        if target_dir:
            stat = target_dir.stat()
            return f"{target_dir}:{stat.st_mtime}"
        else:
            base_stat = self.base_directory_path.stat()
            return f"ALL:{base_stat.st_mtime}"

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEFファイルバッチ処理")

    parser.add_argument(
        "--config_path",
        type=str,
        default="./config/config.yaml",
        help="設定ファイルパス",
    )

    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="最大ワーカ数",
    )

    parser.add_argument(
        "--dir",
        type=str,
        help="処理対象ディレクトリ",
    )

    args = parser.parse_args()

    processor = NEFFileBatchProcess(
        config_path=args.config_path,
        max_workers=args.max_workers,
    )

    if args.dir:
        processor.target_dir = Path(args.dir)
    else:
        processor.target_dir = None

    processor.execute()