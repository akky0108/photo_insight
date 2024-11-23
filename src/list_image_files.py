import os
import csv
import traceback
from datetime import datetime
from typing import List, Dict
from file_handler.exif_file_handler import ExifFileHandler
from batch_framework.base_batch import BaseBatchProcessor

class NEFFileBatchProcess(BaseBatchProcessor):
    """RAWファイルのバッチ処理を行うクラス"""

    def __init__(self, config_path=None, max_workers=4):
        """
        コンストラクタ。設定ファイルの読み込みと初期化を行う。
        
        Args:
            config_path (str): 設定ファイルのパス
            max_workers (int): 最大並列実行数
        """
        super().__init__(config_path=config_path, max_workers=max_workers)
        self.exif_handler = ExifFileHandler()
        self.exif_fields = self.config.get("exif_fields", [
            "FileName", "Model", "Lens", "ISO", "Aperture", "FocalLength",
            "Rating", "ImageHeight", "ImageWidth", "Orientation", "BitDepth"
        ])
        self.append_mode = self.config.get("append_mode", False)
        self.base_directory_path = self.config.get("base_directory_root", "/mnt/l/picture/2024/")
        self.output_directory = self.config.get("output_directory", "temp")
        self.target_date = datetime.strptime(self.config.get("target_date", "2024-05-01"), "%Y-%m-%d")

    def setup(self) -> None:
        """バッチ処理の初期設定を行う。出力先ディレクトリのチェックを実施。"""
        self.temp_dir = os.path.join(self.project_root, self.output_directory)

        if not os.path.isdir(self.base_directory_path):
            error_msg = f"ディレクトリが見つかりません: {self.base_directory_path}"
            self.handle_error(error_msg, raise_exception=True)

        self.logger.info(f"初期設定が完了しました。画像ディレクトリ: {self.base_directory_path}")

    def process(self) -> None:
        """RAWファイルのEXIFデータを取得し、サブディレクトリごとにCSVファイルへ書き出す処理。"""
        try:
            # フックメソッドの呼び出し（BaseBatchProcessor クラスのメソッドを使用）
            if hasattr(super(), 'run_hook'):
                super().run_hook("PRE_PROCESS")
            
            subdirs = self.get_target_subdirectories(self.base_directory_path)
            for subdir in subdirs:
                self.logger.info(f"ディレクトリを処理中: {subdir}")
                self.process_directory(subdir)

            if hasattr(super(), 'run_hook'):
                super().run_hook("POST_PROCESS")

        except Exception as e:
            self.handle_error(f"予期しないエラーが発生しました: {e}", raise_exception=True)

    def cleanup(self) -> None:
        """リソースの解放や後処理を行うメソッド。"""
        self.logger.info("クリーンアップ処理を実行中...")

    def get_target_subdirectories(self, base_path: str) -> List[str]:
        """ターゲット日付以降に作成されたサブディレクトリを取得する。"""
        target_subdirs = []
        for root, dirs, _ in os.walk(base_path):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                dir_creation_time = datetime.fromtimestamp(os.path.getctime(dir_path))
                if dir_creation_time >= self.target_date:
                    target_subdirs.append(dir_path)
                    self.logger.debug(f"ターゲットディレクトリ: {dir_path} (作成日時: {dir_creation_time})")
        return target_subdirs

    def process_directory(self, dir_path: str) -> None:
        """指定したディレクトリ内のRAWファイルを処理してCSVに出力する。"""
        # 対応するRAWファイル形式の拡張子リストを取得
        raw_extensions = self.exif_handler.raw_extensions
        raw_files = self.exif_handler.read_files(dir_path, file_extensions=raw_extensions)

        if not raw_files:
            self.logger.warning(f"ディレクトリ内にRAWファイルが見つかりませんでした: {dir_path}")
            return

        exif_data_list = self.filter_exif_data(raw_files)
        if exif_data_list:
            output_file_path = os.path.join(self.temp_dir, f'{os.path.basename(dir_path)}_raw_exif_data.csv')
            self.write_csv(output_file_path, exif_data_list)

    def filter_exif_data(self, raw_files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """RAWファイルのEXIFデータをフィルタリングしてリストに変換する。"""
        exif_data_list = []
        for raw_file in raw_files:
            # EXIFデータの全内容を確認するためのログ
            self.logger.debug(f"取得したEXIFデータ: {raw_file}")

            filtered_exif_data = {field: raw_file.get(field) for field in self.exif_fields}
            missing_fields = [field for field, value in filtered_exif_data.items() if value is None]
            if missing_fields:
                self.logger.warning(f"不完全なEXIFデータ。欠落フィールド: {missing_fields}")
            exif_data_list.append(filtered_exif_data)
        return exif_data_list

    def write_csv(self, file_path: str, data: List[Dict[str, str]]) -> None:
        """抽出したEXIFデータをCSV形式で指定したファイルに書き込む。"""
        write_mode = 'a' if self.append_mode else 'w'
        try:
            with open(file_path, write_mode, newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.exif_fields)
                if write_mode == 'w':
                    writer.writeheader()
                writer.writerows(data)
            self.logger.info(f"データがCSVファイルに正常に書き込まれました: {file_path}")
        except Exception as e:
            self.handle_error(f"CSVファイルへの書き込み中にエラー: {e}", raise_exception=True)

def main():
    """スクリプトのエントリーポイント"""
    default_config_path = "./config/config.yaml"
    try:
        process = NEFFileBatchProcess(config_path=default_config_path)
        process.execute()
    except Exception as e:
        print(f"バッチ処理中にエラー: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
