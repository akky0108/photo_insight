import os
import csv
import traceback
from datetime import datetime
from typing import List, Dict
from file_handler.exif_file_handler import ExifFileHandler
from batch_framework.base_batch import BaseBatchProcessor

class NEFFileBatchProcess(BaseBatchProcessor):
    """NEFファイルのバッチ処理を行うクラス"""

    def __init__(self, config_path=None):
        """
        コンストラクタ。設定ファイルの読み込みと初期化を行う。
        
        Args:
            config_path (str): 設定ファイルのパス
        """
        super().__init__(config_path=config_path)
        self.exif_handler = ExifFileHandler()  # EXIFデータを扱うハンドラ
        self.exif_fields = self.config.get("exif_fields", [
            "FileName", "Model", "Lens", "ISO", "Aperture", "FocalLength",
            "Rating", "ImageHeight", "ImageWidth", "Orientation", "BitDepth",
            "ExposureBiasValue"
        ])
        self.append_mode = self.config.get("append_mode", False)  # CSVファイルの追記モード設定
        self.base_directory_path = self.config.get("base_directory_root", "/mnt/l/picture/2024/")
        self.output_directory = self.config.get("output_directory", "temp")
        self.target_date = datetime.strptime(self.config.get("target_date", "2024-05-01"), "%Y-%m-%d")  # ターゲット日付を設定

    def setup(self) -> None:
        """
        バッチ処理の初期設定を行う。出力先ディレクトリのチェックを実施。
        """
        self.temp_dir = os.path.join(self.project_root, self.output_directory)

        if not os.path.isdir(self.base_directory_path):
            self.logger.error(f"ディレクトリが見つかりません: {self.base_directory_path}")
            raise FileNotFoundError(f"ディレクトリが見つかりません: {self.base_directory_path}")

        self.logger.info(f"初期設定が完了しました。画像ディレクトリ: {self.base_directory_path}")

    def process(self) -> None:
        """
        NEFファイルのEXIFデータを取得し、サブディレクトリごとにCSVファイルへ書き出す処理。
        ターゲット日付以降に作成されたディレクトリのみを対象とする。
        """
        try:
            subdirs = self.get_target_subdirectories(self.base_directory_path)
            for subdir in subdirs:
                self.logger.info(f"ディレクトリを処理中: {subdir}")
                self.process_directory(subdir)
        except Exception as e:
            self.logger.error(f"予期しないエラーが発生しました: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def get_target_subdirectories(self, base_path: str) -> List[str]:
        """
        ターゲット日付以降に作成されたサブディレクトリを取得する。
        
        Args:
            base_path (str): ベースとなるディレクトリのパス
        
        Returns:
            List[str]: ターゲット日付以降に作成されたサブディレクトリのリスト
        """
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
        """
        指定したディレクトリ内のNEFファイルを処理してCSVに出力する。
        
        Args:
            dir_path (str): 処理対象のディレクトリパス
        """
        nef_files = self.exif_handler.read_files(dir_path)
        if not nef_files:
            self.logger.warning(f"ディレクトリ内に.NEFファイルが見つかりませんでした: {dir_path}")
            return

        exif_data_list = self.filter_exif_data(nef_files)
        if exif_data_list:
            output_file_path = os.path.join(self.temp_dir, f'{os.path.basename(dir_path)}_nef_exif_data.csv')
            self.write_csv(output_file_path, exif_data_list)

    def filter_exif_data(self, nef_files: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        NEFファイルのEXIFデータをフィルタリングしてリストに変換する。
        
        Args:
            nef_files (List[Dict[str, str]]): EXIFデータが含まれるNEFファイルのリスト
        
        Returns:
            List[Dict[str, str]]: フィルタリングされたEXIFデータのリスト
        """
        exif_data_list = []
        for nef_file in nef_files:
            # フィルタリングして必要なEXIFデータのみを抽出
            filtered_exif_data = {field: nef_file.get(field) for field in self.exif_fields}
            missing_fields = [field for field, value in filtered_exif_data.items() if value is None]

            # 欠落しているフィールドがあれば警告ログを出力
            if missing_fields:
                self.logger.warning(f"不完全なEXIFデータ。欠落フィールド: {missing_fields}, ファイル: {filtered_exif_data.get('FileName', '不明')}")
            exif_data_list.append(filtered_exif_data)

        return exif_data_list

    def write_csv(self, file_path: str, data: List[Dict[str, str]]) -> None:
        """
        抽出したEXIFデータをCSV形式で指定したファイルに書き込む。
        
        Args:
            file_path (str): 書き込み先のCSVファイルパス
            data (List[Dict[str, str]]): 書き込むデータのリスト
        """
        write_mode = 'a' if self.append_mode else 'w'
        try:
            with open(file_path, write_mode, newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.exif_fields)
                if write_mode == 'w':
                    writer.writeheader()
                for row in data:
                    writer.writerow(row)
            self.logger.info(f"データがCSVファイルに正常に書き込まれました: {file_path}")
        except Exception as e:
            self.logger.error(f"CSVファイルへの書き込み中にエラーが発生しました: {e}")
            raise

    def cleanup(self) -> None:
        """バッチ処理後のクリーンアップ処理。"""
        self.logger.info("クリーンアップが完了しました。")

def main():
    """スクリプトのエントリーポイント"""
    default_config_path = "./config.yaml"

    try:
        process = NEFFileBatchProcess(config_path=default_config_path)
        process.execute()
    except Exception as e:
        print(f"バッチ処理中にエラーが発生しました: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
