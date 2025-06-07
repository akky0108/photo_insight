import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, List


class FileHandler(ABC):

    plugins = {}
    DEFAULT_FORMAT = "text"  # デフォルトのファイルフォーマットをクラス変数として設定

    @classmethod
    def register_plugin(cls, format, plugin):
        """新しいフォーマットとそのプラグインを登録"""
        cls.plugins[format] = plugin

    @classmethod
    def get_plugin(cls, format):
        """指定されたフォーマットに対応するプラグインを取得"""
        return cls.plugins.get(format)

    def __init__(
        self,
        config: Optional[Dict[str, any]] = None,
        default_format: Optional[str] = None,
    ):
        """
        初期化メソッド。オプションでコンフィグとデフォルトフォーマットを設定可能。
        コンフィグが指定されない場合は空の辞書が使用される。
        デフォルトフォーマットが指定されない場合はクラス変数のDEFAULT_FORMATが使われる。
        """
        self.config = config or {}
        self.default_format = default_format or self.DEFAULT_FORMAT

    @abstractmethod
    def read_file(self, file_path, format=None):
        """ファイルを読み込むための抽象メソッド。フォーマットが指定されない場合はインスタンスのデフォルトフォーマットを使用。"""
        format = format or self.default_format
        pass

    @abstractmethod
    def write_file(self, file_path, data, format=None):
        """ファイルにデータを書き込むための抽象メソッド。フォーマットが指定されない場合はインスタンスのデフォルトフォーマットを使用。"""
        format = format or self.default_format
        pass

    @abstractmethod
    def delete_file(self, file_path):
        """ファイルを削除するための抽象メソッド。"""
        pass

    @abstractmethod
    def update_file(self, file_path, data, format=None):
        """既存のファイルを更新するための抽象メソッド。フォーマットが指定されない場合はインスタンスのデフォルトフォーマットを使用。"""
        format = format or self.default_format
        pass

    def file_exists(self, file_path):
        """指定されたパスにファイルが存在するかを確認する。"""
        return os.path.exists(file_path)

    def get_file_size(self, file_path):
        """
        指定されたファイルのサイズをバイト単位で返す。
        ファイルが存在しない場合は None を返す。
        """
        if self.file_exists(file_path):
            return os.path.getsize(file_path)
        return None

    def validate_format(
        self, file_path: str, expected_formats: Optional[List[str]] = None
    ):
        """
        ファイルの拡張子が期待されるフォーマット（単一または複数）と一致するか確認する。
        expected_formatsがNoneの場合はインスタンスのデフォルトフォーマットと比較。
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip(".")  # 拡張子からピリオドを除去し、小文字に変換
        if expected_formats:
            # 複数のフォーマットと比較
            return ext in [fmt.lower() for fmt in expected_formats]
        # デフォルトフォーマットと比較
        return ext == self.default_format.lower()
