import os
from abc import ABC, abstractmethod
from typing import Optional, Dict

class FileHandler(ABC):

    plugins = {}
    DEFAULT_FORMAT = 'text'  # formatのデフォルト値をクラス変数として設定

    @classmethod
    def register_plugin(cls, format, plugin):
        cls.plugins[format] = plugin

    @classmethod
    def get_plugin(cls, format):
        return cls.plugins.get(format)

    def __init__(self, config: Optional[Dict[str, any]] = None):
        self.config = config or {}

    @abstractmethod
    def read_file(self, file_path, format=DEFAULT_FORMAT):
        pass

    @abstractmethod
    def write_file(self, file_path, data, format=DEFAULT_FORMAT):
        pass

    @abstractmethod
    def delete_file(self, file_path):
        pass

    @abstractmethod
    def update_file(self, file_path, data, format=DEFAULT_FORMAT):
        pass

    def file_exists(self, file_path):
        """指定されたファイルが存在するか確認"""
        return os.path.exists(file_path)

    def get_file_size(self, file_path):
        """指定されたファイルのサイズをバイト単位で返す"""
        if self.file_exists(file_path):
            return os.path.getsize(file_path)
        return None

    def validate_format(self, file_path, expected_format):
        """ファイルの拡張子が期待されるフォーマットと一致するか確認"""
        _, ext = os.path.splitext(file_path)
        return ext.lower() == f".{expected_format.lower()}"