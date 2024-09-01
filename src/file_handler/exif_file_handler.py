import os
import subprocess
import json
from file_handler.file_handler import FileHandler

class ExifFileHandler(FileHandler):

    def read_file(self, file_path, format='exif'):
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")

        if format == 'exif':
            return self.get_exif_data(file_path)
        else:
            plugin = self.get_plugin(format)
            if plugin:
                return plugin.read(file_path)
            raise NotImplementedError(f"No plugin registered for format: {format}")

    def write_file(self, output_file_path, data, format='text', header=None, write_mode='w'):
        plugin = self.get_plugin(format)
        if plugin:
            plugin.write(output_file_path, data, header)
        else:
            raise NotImplementedError(f"No plugin registered for format: {format}")

    def delete_file(self, file_path):
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")
        os.remove(file_path)

    def update_file(self, file_path, data, format='text'):
        self.write_file(file_path, data, format)

    def read_files(self, directory_path, file_extension='.nef'):
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"ディレクトリが存在しません: {directory_path}")

        files = [f for f in os.listdir(directory_path) if f.lower().endswith(file_extension)]
        exif_data_list = []

        for file in files:
            file_path = os.path.join(directory_path, file)
            exif_data = self.get_exif_data(file_path)
            if exif_data:
                exif_data_list.append(exif_data)

        return exif_data_list

    def get_exif_data(self, file_path):
        try:
            result = subprocess.run(
                ['exiftool', '-json', file_path],
                capture_output=True, text=True, check=True
            )
            metadata = json.loads(result.stdout)
            if metadata:
                return metadata[0]
            else:
                raise ValueError("ExifToolの出力が空です")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ExifToolでのプロセスエラー: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONのデコード中にエラーが発生しました: {e}")
        except Exception as e:
            raise RuntimeError(f"予期しないエラーが発生しました: {e}")
