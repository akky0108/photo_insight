import os
import pytest
import logging
from photo_eval_env_manager.merge_envs import (
    merge_envs,
    parse_conda_yaml,
    parse_pip_requirements,
)
from photo_eval_env_manager.envmerge.exceptions import (
    VersionMismatchError,
    DuplicatePackageError,
)
from unittest.mock import Mock

# テストで使用するfixtureファイルのパスを設定
BASE_YML = "tests/fixtures/environment_base.yml"
BASE_YML_DUP = "tests/fixtures/environment_base_with_dup.yml"
BASE_YML_MIS = "tests/fixtures/environment_base_with_mis.yml"
PIP_JSON = "tests/fixtures/pip_list.json"
FINAL_YML = "tests/fixtures/environment_combined.yml"
REQUIREMENTS_TXT = "tests/fixtures/requirements.txt"
CI_YML = "tests/fixtures/environment_ci.yml"
EXCLUDE_CI_TXT = "tests/fixtures/exclude_ci.txt"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
ENV_NAME = "photo_eval_env"


def test_merge_envs_success():
    with open(EXCLUDE_CI_TXT) as f:
        exclude_for_ci = [line.strip() for line in f if line.strip()]

    merge_envs(
        base_yml=BASE_YML,
        pip_json=PIP_JSON,
        final_yml=FINAL_YML,
        requirements_txt=REQUIREMENTS_TXT,
        ci_yml=CI_YML,
        exclude_for_ci=exclude_for_ci,
    )

    assert os.path.exists(FINAL_YML)
    assert os.path.exists(REQUIREMENTS_TXT)

    with open(REQUIREMENTS_TXT) as f:
        content = f.read()
        assert "requests==2.31.0" in content
        assert "flask==2.0.1" in content

    with open(FINAL_YML) as f:
        yml_content = f.read()
        assert "requests==2.31.0" in yml_content

    assert os.path.exists(CI_YML)
    with open(CI_YML) as f:
        ci_content = f.read()
        assert "requests" not in ci_content
        assert "flask==2.0.1" in ci_content


def test_duplicate_package_error():
    base_yml_with_dup = os.path.join(FIXTURES_DIR, "environment_base_with_dup.yml")
    with pytest.raises(DuplicatePackageError):
        merge_envs(
            base_yml=base_yml_with_dup,
            pip_json=PIP_JSON,
            final_yml=FINAL_YML,
            requirements_txt=REQUIREMENTS_TXT,
            strict=True,
        )


def test_base_yml_not_found():
    with pytest.raises(FileNotFoundError):
        merge_envs("non_existent_file.yml", PIP_JSON, FINAL_YML, REQUIREMENTS_TXT)


def test_pip_json_not_found():
    with pytest.raises(FileNotFoundError):
        merge_envs(BASE_YML, "non_existent_pip_list.json", FINAL_YML, REQUIREMENTS_TXT)


def test_version_mismatch_strict():
    with pytest.raises(VersionMismatchError):
        merge_envs(BASE_YML_MIS, PIP_JSON, FINAL_YML, REQUIREMENTS_TXT, strict=True)


def test_cpu_only_version_conversion():
    merge_envs(
        base_yml=BASE_YML,
        pip_json=PIP_JSON,
        final_yml=FINAL_YML,
        requirements_txt=REQUIREMENTS_TXT,
        cpu_only=True,
    )

    with open(REQUIREMENTS_TXT) as f:
        content = f.read()
        assert "torch-cpu==1.9.0" in content


def test_parse_conda_yaml():
    dummy_yaml = {
        "dependencies": ["python=3.10", "numpy", {"pip": ["requests", "scikit-learn"]}]
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


def test_merge_envs_logs_exception_on_version_mismatch():
    mock_logger = Mock(spec=logging.Logger)

    with pytest.raises(VersionMismatchError):
        merge_envs(
            base_yml=BASE_YML_MIS,
            pip_json=PIP_JSON,
            final_yml=FINAL_YML,
            requirements_txt=REQUIREMENTS_TXT,
            strict=True,
            logger=mock_logger,
        )

    mock_logger.exception.assert_called_once()


@pytest.mark.parametrize("pip_path,expected_pkgs", [
    ("tests/fixtures/requirements.txt", ["torch-cpu==1.9.0", "flask==2.0.1"]),
])
def test_merge_envs_with_explicit_pip_format(pip_path, expected_pkgs):
    merge_envs(
        base_yml=BASE_YML,
        pip_json=pip_path,
        final_yml=FINAL_YML,
        requirements_txt=REQUIREMENTS_TXT,
        cpu_only=True,
    )
    with open(REQUIREMENTS_TXT) as f:
        content = f.read()
        for pkg in expected_pkgs:
            assert pkg in content
