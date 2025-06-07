import csv
import time
import argparse
from datetime import datetime
from typing import List, Dict
from pathlib import Path
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

    def setup(self) -> None:
        """バッチ初期化処理（出力ディレクトリ作成など）"""
        super().setup()
        self.temp_dir = Path(self.project_root) / self.output_directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if not self.base_directory_path.is_dir():
            error_msg = f"ディレクトリが見つかりません: {self.base_directory_path}"
            self.handle_error(error_msg, raise_exception=True)

        self.logger.info(f"初期設定完了: 画像ディレクトリ {self.base_directory_path}")

    def process(self) -> None:
        """RAWファイルをスキャンしてEXIFデータをCSV出力"""
        try:
            self.logger.info("RAWファイル処理開始")
            subdirs = self.get_target_subdirectories(self.base_directory_path)
            for subdir in subdirs:
                self.logger.info(f"処理対象ディレクトリ: {subdir}")
                self.process_directory(subdir)
        except Exception as e:
            self.handle_error(
                f"予期しないエラーが発生しました: {e}", raise_exception=True
            )

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

    def process_directory(self, dir_path: Path) -> None:
        """ディレクトリ内のRAWファイルをEXIF抽出してCSV出力"""
        raw_extensions = self.exif_handler.raw_extensions
        raw_files = self.exif_handler.read_files(
            str(dir_path), file_extensions=raw_extensions
        )

        if not raw_files:
            self.logger.warning(f"RAWファイルなし: {dir_path}")
            return

        exif_data_list = self.filter_exif_data(raw_files)
        if exif_data_list:
            output_file_path = self.temp_dir / f"{dir_path.name}_raw_exif_data.csv"
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
        """フィルタしたデータをCSV書き出し (リトライ付き)"""
        write_mode = "a" if self.append_mode else "w"
        for attempt in range(3):
            try:
                with file_path.open(
                    write_mode, newline="", encoding="utf-8"
                ) as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.exif_fields)
                    if write_mode == "w":
                        writer.writeheader()
                    writer.writerows(data)
                self.logger.info(f"CSV出力成功: {file_path}")
                break
            except Exception as e:
                self.logger.error(f"CSV書き込み失敗 ({attempt+1}回目): {e}")
                if attempt == 2:
                    self.handle_error(
                        f"CSVファイル書き込みエラー: {e}", raise_exception=True
                    )
                time.sleep(1)

    def _process_batch(self, batch: List[Path]) -> None:
        """バッチ単位でディレクトリを処理"""
        for dir_path in batch:
            self.logger.info(f"バッチ処理対象: {dir_path}")
            self.process_directory(dir_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEFファイルバッチ処理ツール")
    parser.add_argument(
        "--config_path",
        type=str,
        help="設定ファイルパス (デフォルト: './config/config.yaml')",
    )
    parser.add_argument("--max_workers", type=int, default=4, help="最大ワーカ数")
    args = parser.parse_args()

    processor = NEFFileBatchProcess(
        config_path=args.config_path or "./config/config.yaml",
        max_workers=args.max_workers,
    )
    processor.execute()
