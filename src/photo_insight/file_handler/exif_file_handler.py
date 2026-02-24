import os
import subprocess
import json
from photo_insight.file_handler.file_handler import FileHandler


class ExifFileHandler(FileHandler):
    def __init__(self, raw_extensions=None):
        """コンストラクタ。対応するRAWファイルの拡張子を初期化"""
        super().__init__()
        # デフォルトのRAWファイル拡張子リスト
        self.raw_extensions = raw_extensions or [
            ".NEF",
            ".NRW",
            ".CR2",
            ".CR3",
            ".ARW",
            ".RAF",
            ".RW2",
            ".ORF",
            ".PEF",
        ]

    def read_file(self, file_path, format="exif"):
        """単一のファイルを読み込む"""
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")

        if format == "exif":
            return self.get_exif_data(file_path)
        else:
            plugin = self.get_plugin(format)
            if plugin:
                return plugin.read(file_path)
            raise NotImplementedError(f"No plugin registered for format: {format}")

    def write_file(self, output_file_path, data, format="text", header=None, write_mode="w"):
        """ファイルにデータを書き込む"""
        plugin = self.get_plugin(format)
        if plugin:
            plugin.write(output_file_path, data, header)
        else:
            raise NotImplementedError(f"No plugin registered for format: {format}")

    def delete_file(self, file_path):
        """ファイルを削除する"""
        if not self.file_exists(file_path):
            raise FileNotFoundError(f"ファイルが存在しません: {file_path}")
        os.remove(file_path)

    def update_file(self, file_path, data, format="text"):
        """ファイルを更新する"""
        self.write_file(file_path, data, format)

    def read_files(self, directory_path, file_extensions=None):
        """
        指定したディレクトリ内のRAWファイルのEXIFデータを取得します。

        Args:
            directory_path (str): 読み込み対象のディレクトリパス
            file_extensions (List[str]): 読み込み対象のファイル拡張子リスト

        Returns:
            List[Dict[str, str]]: EXIFデータのリスト
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"ディレクトリが存在しません: {directory_path}")

        # 拡張子リストが指定されていない場合はデフォルトのリストを使用
        if file_extensions is None:
            file_extensions = self.raw_extensions

        # 指定した拡張子のRAWファイルをリストアップ
        files = [
            f for f in os.listdir(directory_path) if any(f.lower().endswith(ext.lower()) for ext in file_extensions)
        ]

        exif_data_list = []
        for file in files:
            file_path = os.path.join(directory_path, file)
            exif_data = self.get_exif_data(file_path)
            if exif_data:
                exif_data_list.append(exif_data)

        return exif_data_list

    def get_exif_data(self, file_path):
        """ExifToolを使用してEXIFデータを取得"""
        try:
            result = subprocess.run(
                ["exiftool", "-json", file_path],
                capture_output=True,
                text=True,
                check=True,
            )
            metadata = json.loads(result.stdout)
            if not metadata:
                raise ValueError("ExifToolの出力が空です")

            # 対象フィールドを数値に変換
            metadata[0] = self.convert_to_numeric(metadata[0])

            # ビット深度の取得
            bit_depth = self.get_bit_depth(metadata[0])
            metadata[0]["BitDepth"] = bit_depth

            return metadata[0]
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ExifToolでのプロセスエラー: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONのデコード中にエラーが発生しました: {e}")
        except Exception as e:
            raise RuntimeError(f"予期しないエラーが発生しました: {e}")

    def get_bit_depth(self, metadata):
        """ビット深度を取得"""
        if "BitsPerSample" in metadata:
            return metadata["BitsPerSample"]
        elif "BitDepth" in metadata:
            return metadata["BitDepth"]
        else:
            return "ビット深度情報が見つかりません"

    def convert_to_numeric(self, metadata):
        """数値変換を行う"""
        if "ISO" in metadata:
            metadata["ISO"] = self.convert_iso(metadata["ISO"])

        if "Aperture" in metadata:
            metadata["Aperture"] = self.convert_aperture(metadata["Aperture"])

        if "FocalLength" in metadata:
            metadata["FocalLength"] = self.convert_focal_length(metadata["FocalLength"])

        if "Orientation" in metadata:
            metadata["Orientation"] = self.convert_orientation(metadata["Orientation"])

        return metadata

    def convert_iso(self, iso_value):
        """ISOを数値に変換"""
        try:
            return int(iso_value)
        except ValueError:
            return None

    def convert_aperture(self, aperture_value):
        """Apertureを数値に変換"""
        try:
            if isinstance(aperture_value, str) and aperture_value.startswith("f/"):
                return float(aperture_value[2:])
            return float(aperture_value)
        except ValueError:
            return None

    def convert_focal_length(self, focal_length_value):
        """FocalLengthを数値に変換"""
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
        }
        return conversion_map.get(orientation_value, None)
