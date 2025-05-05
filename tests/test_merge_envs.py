import os
import pytest
from unittest.mock import patch
from photo_eval_env_manager.merge_envs import merge_envs
from photo_eval_env_manager.envmerge.exceptions import DuplicatePackageError, VersionMismatchError

# mock データ用のパス設定
BASE_YML = 'tests/fixtures/environment_base.yml'
PIP_JSON = 'tests/fixtures/pip_list.json'
FINAL_YML = 'tests/fixtures/environment_combined.yml'
REQUIREMENTS_TXT = 'tests/fixtures/requirements.txt'
CI_YML = "tests/fixtures/environment_ci.yml"
EXCLUDE_CI_TXT = "tests/fixtures/exclude_ci.txt"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

# 正常系: merge_envsが正常に動作するか確認
def test_merge_envs_success():
    # exclude list をセット
    with open(EXCLUDE_CI_TXT) as f:
        exclude_for_ci = [line.strip() for line in f if line.strip()]

    # 実行
    merge_envs(
        base_yml=BASE_YML,
        pip_json=PIP_JSON,
        final_yml=FINAL_YML,
        requirements_txt=REQUIREMENTS_TXT,
        ci_yml=CI_YML,
        exclude_for_ci=exclude_for_ci
    )

    # 検証: 通常環境
    assert os.path.exists(FINAL_YML)
    assert os.path.exists(REQUIREMENTS_TXT)

    with open(REQUIREMENTS_TXT) as f:
        content = f.read()
        assert "requests==2.31.0" in content
        assert "flask==2.0.1" in content

    with open(FINAL_YML) as f:
        yml_content = f.read()
        assert "requests==2.31.0" in yml_content

    # 検証: CI 環境
    assert os.path.exists(CI_YML)
    with open(CI_YML) as f:
        ci_content = f.read()
        assert "requests" not in ci_content
        assert "flask==2.0.1" in ci_content

# 異常系: Base YAMLがない場合の挙動
def test_base_yml_not_found():
    with pytest.raises(FileNotFoundError):
        merge_envs('non_existent_file.yml', PIP_JSON, FINAL_YML, REQUIREMENTS_TXT)


# 異常系: pip list JSON がない場合の挙動
def test_pip_json_not_found():
    with pytest.raises(FileNotFoundError):
        merge_envs(BASE_YML, 'non_existent_pip_list.json', FINAL_YML, REQUIREMENTS_TXT)


# 異常系: 依存パッケージのバージョン不一致（Strictモード）
def test_version_mismatch_strict(capfd):
    with pytest.raises(VersionMismatchError):
        merge_envs(BASE_YML, PIP_JSON, FINAL_YML, REQUIREMENTS_TXT, strict=True)


# 異常系: 重複パッケージが存在する場合
def test_duplicate_package_error():
    base_yml_with_dup = os.path.join(FIXTURES_DIR, "environment_base_with_dup.yml")
    with pytest.raises(DuplicatePackageError):
        merge_envs(
            base_yml=base_yml_with_dup,
            pip_json=PIP_JSON,
            final_yml=FINAL_YML,
            requirements_txt=REQUIREMENTS_TXT
        )