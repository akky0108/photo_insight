import pytest
from photo_eval_env_manager.merge_envs import parse_pip_package, validate_version_string

def test_parse_pip_package_simple():
    assert parse_pip_package("numpy==1.23.5") == "numpy"

def test_parse_pip_package_url():
    assert parse_pip_package("my-package @ https://example.com/pkg.whl") == "my-package"

def test_validate_version_string_valid():
    assert validate_version_string("python=3.10")
    assert validate_version_string("numpy>=1.20")
    assert validate_version_string("pandas<=1.4")

def test_validate_version_string_invalid():
    assert not validate_version_string("python=3.1,=3.10.*")  # 不正な複数バージョン指定
    assert not validate_version_string("invalid==")           # 値なし
