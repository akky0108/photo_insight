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
            # すべてのEXIFデータを取得
            result = subprocess.run(
                ['exiftool', '-json', file_path],
                capture_output=True, text=True, check=True
            )
            metadata = json.loads(result.stdout)
            if not metadata:
                raise ValueError("ExifToolの出力が空です")

            # 対象フィールドを数値に変換する
            metadata[0] = self.convert_to_numeric(metadata[0])

            return metadata[0]
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ExifToolでのプロセスエラー: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONのデコード中にエラーが発生しました: {e}")
        except Exception as e:
            raise RuntimeError(f"予期しないエラーが発生しました: {e}")

    def convert_to_numeric(self, metadata):
        """ISO, Aperture, FocalLength, Orientationを数値に変換する関数"""

        # ISO の数値変換
        if 'ISO' in metadata:
            metadata['ISO'] = self.convert_iso(metadata['ISO'])

        # Aperture の数値変換
        if 'Aperture' in metadata:
            metadata['Aperture'] = self.convert_aperture(metadata['Aperture'])

        # FocalLength の数値変換
        if 'FocalLength' in metadata:
            metadata['FocalLength'] = self.convert_focal_length(metadata['FocalLength'])

        # Orientation の数値変換
        if 'Orientation' in metadata:
            metadata['Orientation'] = self.convert_orientation(metadata['Orientation'])

        return metadata

    def convert_iso(self, iso_value):
        """ISOを数値に変換"""
        try:
            return int(iso_value)
        except ValueError:
            return None

    def convert_aperture(self, aperture_value):
        """Aperture（例: f/2.8 形式）を数値に変換"""
        try:
            if isinstance(aperture_value, str) and aperture_value.startswith("f/"):
                return float(aperture_value[2:])
            return float(aperture_value)
        except ValueError:
            return None

    def convert_focal_length(self, focal_length_value):
        """FocalLength（例: 50mm 形式）を数値に変換"""
        try:
            if isinstance(focal_length_value, str) and focal_length_value.endswith("mm"):
                return float(focal_length_value[:-2])
            return float(focal_length_value)
        except ValueError:
            return None

    def convert_orientation(self, orientation_value):
        """Orientationを数値に変換"""
        conversion_map = {
            "Horizontal (normal)": 1,
            "Rotate 90 CW": 6,
            "Rotate 270 CW": 8,
            "Rotate 180": 3,
            # 他の変換も追加可能
        }
        return conversion_map.get(orientation_value, None)