
import argparse
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

       # Base契約: setup() で self.data が作られる（load_data() が呼ばれる）
        self.target_dirs: List[Path] = []

    # ------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------
    def setup(self) -> None:
        # ★ 必須：常に定義（Base.setup() から load_data() が呼ばれるため）
        self.target_dirs = []

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

        # Base契約: setup() -> self.data = self.get_data() -> after_data_loaded(self.data)
        super().setup()

    def cleanup(self) -> None:
        super().cleanup()
        self.logger.info("クリーンアップ完了")

    # ------------------------------------------------------------
    # data collection
    # ------------------------------------------------------------
    def load_data(self) -> List[Dict[str, Any]]:
        """
        BaseBatchProcessor の新契約:
        - load_data(): 純I/O（副作用なし）
        - キャッシュは Base が握る（get_data() は Base 側）
       
        仕様:
        - CLI などで self.target_dir が設定されていれば、そのディレクトリのみ処理
        - 未設定なら setup() で収集した self.target_dirs を全対象に処理
        """
        target_dir: Optional[Path] = getattr(self, "target_dir", None)
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

        data = [
            {
                "path": path,
                "subdir_name": path.parent.name,
            }
            for path in nef_files
        ]

        self.logger.info(f"get_data(): 収集ファイル数 = {len(data)}")
        return data

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