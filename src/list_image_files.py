import os
import csv
from dotenv import load_dotenv
from file_handler.exif_file_handler import ExifFileHandler
from log_util import Logger  # ログクラスをインポート
from batch_framework.base_batch import BaseBatchProcessor

class NEFFileBatchProcess(BaseBatchProcessor):
    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.directory_path = None
        self.exif_handler = ExifFileHandler()
        self.temp_dir = None
        self.output_file_path = None
        # EXIFフィールドにF値（Aperture）とISO感度を追加
        self.exif_fields = ["FileName", "Model", "Lens", "ExposureTime", "FocalLength", "ShutterSpeed", "Aperture", "ISO", "Rating"]
        self.append_mode = False  # 既存ファイルに追記するかどうか

    def setup(self):
        """バッチ処理の初期設定"""
        load_dotenv()
        project_root = os.getenv('PROJECT_ROOT')

        if project_root is None:
            self.logger.error("環境変数 'PROJECT_ROOT' が設定されていません。")
            raise EnvironmentError("環境変数 'PROJECT_ROOT' が設定されていません。")

        self.temp_dir = os.path.join(project_root, 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)

        self.output_file_path = os.path.join(self.temp_dir, 'nef_exif_data.csv')

        if self.directory_path is None:
            self.directory_path = '/mnt/l/picture/2024/2024-07-02'  # デフォルトのディレクトリパス

        self.logger.info("初期設定が完了しました。")

    def process(self):
        """バッチ処理のメインロジック"""
        try:
            nef_files = self.exif_handler.read_files(self.directory_path)
            if not nef_files:
                self.logger.info(f"指定したディレクトリに.NEFファイルが見つかりませんでした: {self.directory_path}")
                print("指定したディレクトリに.NEFファイルが見つかりませんでした。")
            else:
                exif_data_list = []
                for nef_file in nef_files:
                    filtered_exif_data = {field: nef_file.get(field, "N/A") for field in self.exif_fields}
                    exif_data_list.append(filtered_exif_data)
                
                self.write_csv(self.output_file_path, exif_data_list)
                self.logger.info(f"{len(exif_data_list)} 個の.NEFファイルのEXIFデータを {self.output_file_path} に出力しました。")
                print(f"{len(exif_data_list)} 個の.NEFファイルのEXIFデータを {self.output_file_path} に出力しました。")
        except FileNotFoundError:
            self.logger.error(f"ディレクトリが見つかりません: {self.directory_path}")
            print(f"エラー: 指定したディレクトリが見つかりませんでした: {self.directory_path}")
            raise
        except Exception as e:
            self.logger.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
            print(f"エラー: ファイルの読み込み中に問題が発生しました。")
            raise

    def write_csv(self, file_path, data):
        """EXIFデータをCSV形式で書き込む"""
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

    def cleanup(self):
        """バッチ処理後のクリーンアップ"""
        self.logger.info("クリーンアップが完了しました。")

def main():
    process = NEFFileBatchProcess()
    #process.directory_path = '/mnt/l/picture/2024/2024-07-02'  # 必要ならここでディレクトリを指定
    process.directory_path = '/mnt/l/picture/2024/2024-08-26'  # 必要ならここでディレクトリを指定
    process.execute()

if __name__ == "__main__":
    main()
