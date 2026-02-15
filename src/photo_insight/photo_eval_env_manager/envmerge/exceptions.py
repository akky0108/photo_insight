# exceptions.py


class EnvMergeError(Exception):
    """ベースとなる例外クラス"""

    pass


class InvalidVersionError(EnvMergeError):
    """無効なバージョン文字列のときに発生"""

    def __init__(self, version: str):
        super().__init__(f"Invalid version string: '{version}'")
        self.version = version


class DuplicatePackageError(EnvMergeError):
    """複数の異なるバージョンのパッケージが検出されたとき"""

    def __init__(self, package: str, versions: list[str]):
        versions_str = ", ".join(versions)
        super().__init__(f"Duplicate versions for package '{package}': {versions_str}")


class VersionMismatchError(EnvMergeError):
    """conda と pip のバージョン不一致時に発生"""

    def __init__(self, message: str):
        super().__init__(f"Version mismatch detected:\n{message}")
