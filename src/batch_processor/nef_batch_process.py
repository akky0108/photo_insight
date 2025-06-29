import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from threading import Lock
from collections import defaultdict
from typing import Optional
from file_handler.exif_file_handler import ExifFileHandler
from batch_framework.base_batch import BaseBatchProcessor

ExifData = Dict[str, str]


class NEFFileBatchProcess(BaseBatchProcessor):
    """RAWファイルのバッチ処理を行うクラス"""

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
            self.config.get("base_directory_root", "/mnt/l/picture/2024/")
        )
        self.output_directory = self.config.get("output_directory", "temp")
        self.target_date = datetime.strptime(
            self.config.get("target_date", "2024-05-01"), "%Y-%m-%d"
        )
        self._csv_locks: Dict[str, Lock] = defaultdict(Lock)  # ← これを追加

    def _get_lock_for_file(self, file_path: Path) -> Lock:
        key = str(file_path.resolve())
        if key not in self._csv_locks:
            self._csv_locks[key] = Lock()
        return self._csv_locks[key]

    def setup(self) -> None:
        """バッチ初期化処理（出力ディレクトリ作成など）"""
        super().setup()
        self.temp_dir = Path(self.project_root) / self.output_directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if not self.base_directory_path.is_dir():
            error_msg = f"ディレクトリが見つかりません: {self.base_directory_path}"
            self.handle_error(error_msg, raise_exception=True)

        self.logger.info(f"初期設定完了: 画像ディレクトリ {self.base_directory_path}")

    def cleanup(self) -> None:
        """リソースの解放など後処理"""
        super().cleanup()
        self.logger.info("クリーンアップ完了")

    def get_target_subdirectories(self, base_path: Path, depth: int = 1) -> List[Path]:
        """
        ターゲット日付以降に作成されたサブディレクトリを取得

        Args:
            base_path (Path): ベースパス
            depth (int): 探索深さ (1なら直下のみ)

        Returns:
            List[Path]: 対象サブディレクトリのリスト
        """
        target_subdirs = []
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                # サブディレクトリ内のファイルの最終更新時間を確認
                latest_mtime = max(
                    (f.stat().st_mtime for f in subdir.glob("**/*") if f.is_file()),
                    default=subdir.stat().st_mtime,
                )
                if datetime.fromtimestamp(latest_mtime) >= self.target_date:
                    target_subdirs.append(subdir)
        return target_subdirs

    def process_directory(self, subdir_name: str, raw_files: List[Dict]) -> None:
        """
        単一ディレクトリのNEFファイル群を処理してCSVに出力する

        Args:
            subdir_name (str): サブディレクトリ名（出力ファイル名の一部に使用）
            raw_files (List[Dict]): そのディレクトリのNEFファイルに関するEXIF辞書群
        """
        if not raw_files:
            self.logger.warning(f"[単体処理] 対象ファイルなし: {subdir_name}")
            return

        self.logger.info(f"[{subdir_name}] EXIF抽出開始: {len(raw_files)}件")
        exif_data_list = self.filter_exif_data(raw_files)

        if not exif_data_list:
            self.logger.warning(f"[{subdir_name}] EXIF抽出結果なし")
            return

        output_file_path = self.temp_dir / f"{subdir_name}_raw_exif_data.csv"
        self.write_csv(output_file_path, exif_data_list)

    def filter_exif_data(self, raw_files: List[ExifData]) -> List[ExifData]:
        """EXIFフィールドを抽出・フィルタリング"""
        exif_data_list = []
        for raw_file in raw_files:
            self.logger.debug(f"取得EXIFデータ: {raw_file}")
            filtered_exif = {
                field: raw_file.get(field, "N/A") for field in self.exif_fields
            }
            missing_fields = [
                field for field, value in filtered_exif.items() if value == "N/A"
            ]
            if missing_fields:
                self.logger.warning(f"EXIFデータ欠損: {missing_fields}")
            exif_data_list.append(filtered_exif)
        return exif_data_list

    def write_csv(self, file_path: Path, data: List[ExifData]) -> None:
        """フィルタしたデータをCSV書き出し (リトライ + 排他制御付き)"""
        write_mode = "a" if self.append_mode else "w"
        lock = self._get_lock_for_file(file_path)

        for attempt in range(3):
            try:
                is_new_file = not file_path.exists()
                with lock:
                    with file_path.open(write_mode, newline="", encoding="utf-8") as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=self.exif_fields)
                        if write_mode == "w" or is_new_file:
                            writer.writeheader()
                        writer.writerows(data)
                self.logger.info(f"CSV出力成功: {file_path}")
                break
            except Exception as e:
                self.logger.error(f"CSV書き込み失敗 ({attempt+1}回目): {e}")
                if attempt == 2:
                    self.handle_error(f"CSVファイル書き込みエラー: {e}", raise_exception=True)
                time.sleep(1)

    def get_data(self) -> List[Dict]:
        """
        対象ディレクトリ以下の NEF ファイルをスキャンし、EXIF 抽出前のデータを Dict で返却

        Returns:
            List[Dict]: 処理対象の NEF ファイル情報リスト
        """
        raw_extensions = self.exif_handler.raw_extensions
        subdirs = self.get_target_subdirectories(self.base_directory_path)

        results: List[Dict] = []

        for subdir in subdirs:
            raw_files = self.exif_handler.read_files(
                str(subdir), file_extensions=raw_extensions
            )

            for file_data in raw_files:
                file_path = file_data.get("SourceFile")
                if not file_path:
                    continue  # 異常データをスキップ

                result = {
                    "path": file_path,
                    "directory": str(subdir),
                    "filename": Path(file_path).name,
                    "subdir_name": subdir.name,
                    "exif_raw": file_data,
                }
                results.append(result)

        self.logger.info(f"スキャン完了: 対象ファイル数={len(results)}")
        return results

    def _process_batch(self, batch: List[Dict]) -> None:
        """バッチ単位で NEF ファイルを処理"""
        grouped = defaultdict(list)
        for item in batch:
            subdir_name = item.get("subdir_name", "unknown")
            exif_raw = item.get("exif_raw", {})
            grouped[subdir_name].append(exif_raw)

        for subdir_name, raw_list in grouped.items():
            self.logger.info(f"[バッチ] CSV書き出し: {subdir_name} ({len(raw_list)}件)")
            exif_data_list = self.filter_exif_data(raw_list)
            output_file_path = self.temp_dir / f"{subdir_name}_raw_exif_data.csv"
            self.write_csv(output_file_path, exif_data_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEFファイルバッチ処理ツール")
    parser.add_argument(
        "--config_path",
        type=str,
        help="設定ファイルパス (デフォルト: './config/config.yaml')",
    )
    parser.add_argument("--max_workers", type=int, default=4, help="最大ワーカ数")
    parser.add_argument("--dir", type=str, help="単一ディレクトリ処理（デバッグ・再実行用）")
    args = parser.parse_args()

    processor = NEFFileBatchProcess(
        config_path=args.config_path or "./config/config.yaml",
        max_workers=args.max_workers,
    )

    if args.dir:
        from file_handler.exif_file_handler import ExifFileHandler  # 念のため再import安全

        target_dir = Path(args.dir)
        if not target_dir.exists() or not target_dir.is_dir():
            processor.handle_error(f"指定ディレクトリが存在しません: {target_dir}", raise_exception=True)

        raw_extensions = processor.exif_handler.raw_extensions
        raw_files = processor.exif_handler.read_files(
            str(target_dir), file_extensions=raw_extensions
        )
        processor.setup()  # 初期化（temp_dir 等）
        processor.process_directory(target_dir.name, raw_files)
        processor.cleanup()
    else:
        processor.execute()

