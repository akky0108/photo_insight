# csv_file_handler.py
from file_handler.file_handler import FileHandler  # 抽象クラスをインポート
import os
import csv


class CSVPlugin:
    @staticmethod
    def read(file_path):
        with open(file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)

    @staticmethod
    def write(file_path, data, header=None, mode="w"):
        with open(file_path, mode=mode, newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=header or data[0].keys())
            if mode == "w":
                writer.writeheader()
            writer.writerows(data)


class CSVFileHandler(FileHandler):
    def __init__(self, config=None):
        super().__init__(config)
        self.plugin = self.get_plugin("csv")

    def read_file(
        self, file_path, format="csv", filters=None, sort_key=None, reverse=False
    ):
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")

        if format == "csv":
            data = self.plugin.read(file_path)

            # フィルタリング
            if filters:
                data = [
                    row for row in data if all(row[k] == v for k, v in filters.items())
                ]

            # 並び替え
            if sort_key:
                data.sort(key=lambda x: x.get(sort_key, ""), reverse=reverse)

            return data
        else:
            raise NotImplementedError(f"未対応のフォーマットです: {format}")

    def write_file(self, file_path, data, format="csv", header=None, write_mode="w"):
        if format == "csv":
            mode = write_mode if write_mode in ["w", "a"] else "w"
            self.plugin.write(file_path, data, header, mode)
        else:
            raise NotImplementedError(f"未対応のフォーマットです: {format}")

    def delete_file(self, file_path):
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")
        os.remove(file_path)

    def update_file(self, file_path, data, format="csv"):
        self.write_file(file_path, data, format)


# Register the CSV plugin
FileHandler.register_plugin("csv", CSVPlugin())
