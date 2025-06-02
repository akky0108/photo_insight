import os
import pytest
from unittest.mock import patch
from photo_eval_env_manager.merge_envs import merge_envs, parse_conda_yaml, parse_pip_requirements, build_merged_env_dict
from photo_eval_env_manager.envmerge.exceptions import DuplicatePackageError, VersionMismatchError

# テストで使用するfixtureファイルのパスを設定
BASE_YML = 'tests/fixtures/environment_base.yml'
BASE_YML_MIS = 'tests/fixtures/environment_base_with_dup.yml'
PIP_JSON = 'tests/fixtures/pip_list.json'
FINAL_YML = 'tests/fixtures/environment_combined.yml'
REQUIREMENTS_TXT = 'tests/fixtures/requirements.txt'
CI_YML = "tests/fixtures/environment_ci.yml"
EXCLUDE_CI_TXT = "tests/fixtures/exclude_ci.txt"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
ENV_NAME = "photo_eval_env"

def test_merge_envs_success():
    """
    正常系テスト: merge_envs関数が正しく環境ファイルをマージできるか検証する。

    - exclude_ci.txtに記載されたパッケージはCI環境用YAMLに含まれないことを確認。
    - 通常の環境YAML・requirements.txtファイルに期待されるパッケージが含まれていることを検証。
    """
    # CI除外リストをファイルから読み込み
    with open(EXCLUDE_CI_TXT) as f:
        exclude_for_ci = [line.strip() for line in f if line.strip()]

    # merge_envsを実行
    merge_envs(
        base_yml=BASE_YML,
        pip_json=PIP_JSON,
        final_yml=FINAL_YML,
        requirements_txt=REQUIREMENTS_TXT,
        ci_yml=CI_YML,
        exclude_for_ci=exclude_for_ci
    )

    # 出力ファイルが存在することを検証
    assert os.path.exists(FINAL_YML)
    assert os.path.exists(REQUIREMENTS_TXT)

    # requirements.txtに特定パッケージが含まれていることを検証
    with open(REQUIREMENTS_TXT) as f:
        content = f.read()
        assert "requests==2.31.0" in content
        assert "flask==2.0.1" in content

    # 環境YAMLにも同様のパッケージが含まれていることを検証
    with open(FINAL_YML) as f:
        yml_content = f.read()
        assert "requests==2.31.0" in yml_content

    # CI環境用YAMLの検証
    assert os.path.exists(CI_YML)
    with open(CI_YML) as f:
        ci_content = f.read()
        # excludeリストにあるrequestsは含まれないこと
        assert "requests" not in ci_content
        # flaskは含まれていること
        assert "flask==2.0.1" in ci_content


def test_base_yml_not_found():
    """
    異常系テスト: ベースYAMLファイルが存在しない場合にFileNotFoundErrorが発生することを検証。
    """
    with pytest.raises(FileNotFoundError):
        merge_envs('non_existent_file.yml', PIP_JSON, FINAL_YML, REQUIREMENTS_TXT)


def test_pip_json_not_found():
    """
    異常系テスト: pipリストJSONファイルが存在しない場合にFileNotFoundErrorが発生することを検証。
    """
    with pytest.raises(FileNotFoundError):
        merge_envs(BASE_YML, 'non_existent_pip_list.json', FINAL_YML, REQUIREMENTS_TXT)


def test_version_mismatch_strict(capfd):
    """
    異常系テスト: strict=Trueモードでバージョン不一致がある場合にVersionMismatchErrorが発生することを検証。
    """
    with pytest.raises(VersionMismatchError):
        merge_envs(BASE_YML_MIS, PIP_JSON, FINAL_YML, REQUIREMENTS_TXT, strict=True)


def test_duplicate_package_error():
    """
    異常系テスト: ベースYAMLに重複パッケージが存在する場合にDuplicatePackageErrorが発生することを検証。
    """
    base_yml_with_dup = os.path.join(FIXTURES_DIR, "environment_base_with_dup.yml")
    with pytest.raises(DuplicatePackageError):
        merge_envs(
            base_yml=base_yml_with_dup,
            pip_json=PIP_JSON,
            final_yml=FINAL_YML,
            requirements_txt=REQUIREMENTS_TXT
        )


def test_cpu_only_version_conversion():
    """
    異常系テスト: cpu_only=TrueオプションでtorchのバージョンがCPU専用版に変換されることを検証。
    """
    merge_envs(
        base_yml=BASE_YML,
        pip_json=PIP_JSON,
        final_yml=FINAL_YML,
        requirements_txt=REQUIREMENTS_TXT,
        cpu_only=True
    )

    # torchがGPU版からCPU版に変換されていることをrequirements.txtから確認
    with open(REQUIREMENTS_TXT) as f:
        content = f.read()
        assert "torch==1.9.0" in content  # GPU付きバージョンからCPUバージョンに変換済み

def test_parse_conda_yaml():
    dummy_yaml = {
        "dependencies": [
            "python=3.10",
            "numpy",
            {"pip": ["requests", "scikit-learn"]}
        ]
    }
    conda, pip = parse_conda_yaml(dummy_yaml)
    assert conda == ["python=3.10", "numpy"]
    assert pip == ["requests", "scikit-learn"]

def test_parse_pip_requirements_json():
    json_input = '[{"name": "numpy", "version": "1.24.1"}, {"name": "pandas", "version": "1.3.5"}]'
    expected = ["numpy==1.24.1", "pandas==1.3.5"]
    assert parse_pip_requirements(json_input) == expected

def test_parse_pip_requirements_text():
    text_input = "torch==2.0.0\n# コメント行\nscikit-learn==1.2.1"
    expected = ["torch==2.0.0", "scikit-learn==1.2.1"]
    assert parse_pip_requirements(text_input) == expected

def test_build_merged_env_dict():
    conda = ["python=3.10", "numpy=1.24.0", {"pip": ["some-old-thing==1.0.0"]}]
    pip = ["torch==2.0.1", "scikit-learn==1.3.0"]
    env_dict = build_merged_env_dict(conda, pip)

    assert env_dict["name"] == ENV_NAME
    assert {"pip": pip} in env_dict["dependencies"]
    assert not any(isinstance(dep, dict) and "pip" in dep and dep["pip"] == ["some-old-thing==1.0.0"]
                   for dep in env_dict["dependencies"][:-1])  # pipセクションは最後のみにあること
