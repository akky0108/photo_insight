# env_constants.py

# デフォルト Python バージョン
DEFAULT_PYTHON_VERSION = "3.10"

# GPU → CPU パッケージ置換マッピング
GPU_PACKAGE_REPLACEMENTS = {
    "torch": "torch-cpu",
    "tensorflow": "tensorflow-cpu",
    "pytorch": "pytorch-cpu",
    "cudatoolkit": None,
    "pytorch-cuda": None,
}


def is_gpu_package(pkg_name: str) -> bool:
    """
    指定されたパッケージ名が GPU 専用パッケージかどうかを判定する。

    Args:
        pkg_name (str): パッケージ名（小文字前提）

    Returns:
        bool: GPU専用パッケージであれば True
    """
    return pkg_name in GPU_PACKAGE_REPLACEMENTS
