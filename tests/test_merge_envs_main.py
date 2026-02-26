import pytest
import sys
from types import SimpleNamespace
from unittest import mock
from photo_insight.photo_eval_env_manager import merge_envs as target_module
from photo_insight.photo_eval_env_manager.merge_envs import run_cli
from photo_insight.core.logging import AppLogger
import photo_insight.photo_eval_env_manager.merge_envs as main_module


def test_main_logs_exception_and_exits(monkeypatch):
    # モック logger
    mock_logger = mock.Mock()

    # merge_envs 関数を例外を投げるように置き換える
    monkeypatch.setattr(target_module, "merge_envs", mock.Mock(side_effect=RuntimeError("Mocked error")))

    # parse_args の戻り値をモック
    mock_args = mock.Mock(
        conda="dummy_conda.yml",
        pip="dummy_pip.txt",
        output="dummy_output.yml",
        strict=False,
        cpu_only=False,
        log_level="DEBUG",
    )
    monkeypatch.setattr(target_module, "parse_args", mock.Mock(return_value=mock_args))

    # AppLogger.get_logger() がモック logger を返すように
    mock_app_logger = mock.Mock()
    mock_app_logger.get_logger.return_value = mock_logger
    monkeypatch.setattr(target_module, "AppLogger", mock.Mock(return_value=mock_app_logger))

    # sys.exit をモック
    monkeypatch.setattr(sys, "exit", mock.Mock())

    # main() を実行
    target_module.main()

    # logger.exception が呼ばれたことを確認
    mock_logger.exception.assert_called_once_with("UnHandled exception occurred")
    # sys.exit(1) が呼ばれたことを確認
    sys.exit.assert_called_once_with(1)


@pytest.fixture
def logger():
    return AppLogger(project_root=".", logger_name="test_envmerge").get_logger()


def test_run_cli_basic(tmp_path, logger):
    # テスト用の仮ファイル
    conda_path = tmp_path / "env.yml"
    pip_path = tmp_path / "req.txt"
    output_path = tmp_path / "merged.yml"

    conda_path.write_text("""
name: test-env
dependencies:
  - numpy=1.21.0
""")

    pip_path.write_text("requests==2.31.0")

    args = SimpleNamespace(
        conda=conda_path,
        pip=pip_path,
        output=output_path,
        strict=False,
        cpu_only=False,
        log_level="INFO",
    )

    run_cli(args, logger)

    assert output_path.exists()
    assert "numpy=1.21.0" in output_path.read_text()


def test_run_cli_missing_pip_file(tmp_path, logger):
    conda_path = tmp_path / "env.yml"
    pip_path = tmp_path / "missing.txt"
    conda_path.write_text("name: dummy\ndependencies:\n  - numpy")

    args = SimpleNamespace(
        conda=conda_path,
        pip=pip_path,  # 存在しないファイル
        output=tmp_path / "merged.yml",
        strict=False,
        cpu_only=False,
        log_level="INFO",
    )

    with pytest.raises(FileNotFoundError):
        run_cli(args, logger)


def test_run_cli_strict_mode_conflict(tmp_path, logger):
    conda_path = tmp_path / "env.yml"
    pip_path = tmp_path / "req.txt"
    conda_path.write_text("name: test-env\ndependencies:\n  - numpy==1.21.0")
    pip_path.write_text("numpy==1.21.0")

    args = SimpleNamespace(
        conda=conda_path,
        pip=pip_path,
        output=tmp_path / "merged.yml",
        strict=True,
        cpu_only=False,
        log_level="INFO",
    )

    from photo_insight.photo_eval_env_manager.envmerge.exceptions import (
        VersionMismatchError,
    )

    with pytest.raises(VersionMismatchError):
        run_cli(args, logger)


def test_run_cli_duplicate_version_conflict(tmp_path, logger):
    conda_path = tmp_path / "env.yml"
    pip_path = tmp_path / "req.txt"
    conda_path.write_text("name: test-env\ndependencies:\n  - numpy==1.21.0")
    pip_path.write_text("numpy==1.22.0")

    args = SimpleNamespace(
        conda=conda_path,
        pip=pip_path,
        output=tmp_path / "merged.yml",
        strict=True,
        cpu_only=False,
        log_level="INFO",
    )

    from photo_insight.photo_eval_env_manager.envmerge.exceptions import (
        VersionMismatchError,
    )

    with pytest.raises(VersionMismatchError):
        run_cli(args, logger)


def test_run_cli_broken_yaml(tmp_path, logger):
    conda_path = tmp_path / "broken.yml"
    pip_path = tmp_path / "req.txt"
    conda_path.write_text("dependencies:\n  - numpy\n  - invalid: [")
    pip_path.write_text("requests==2.31.0")

    args = SimpleNamespace(
        conda=conda_path,
        pip=pip_path,
        output=tmp_path / "merged.yml",
        strict=False,
        cpu_only=False,
        log_level="INFO",
    )

    import yaml

    with pytest.raises(yaml.YAMLError):
        run_cli(args, logger)


def test_run_cli_unhandled_exception(monkeypatch, tmp_path, logger):
    conda_path = tmp_path / "env.yml"
    pip_path = tmp_path / "req.txt"
    conda_path.write_text("name: test-env\ndependencies:\n  - numpy")
    pip_path.write_text("requests==2.31.0")

    args = SimpleNamespace(
        conda=conda_path,
        pip=pip_path,
        output=tmp_path / "merged.yml",
        strict=False,
        cpu_only=False,
        log_level="INFO",
    )

    # run_cli() を定義しているモジュールのスコープに patch する
    monkeypatch.setattr(
        main_module,
        "load_yaml_file",
        lambda path: (_ for _ in ()).throw(RuntimeError("Test error")),
    )

    with pytest.raises(RuntimeError, match="Test error"):
        main_module.run_cli(args, logger)


def test_run_cli_with_pip_format_txt(tmp_path, logger):
    conda_path = tmp_path / "env.yml"
    pip_path = tmp_path / "req.txt"
    output_path = tmp_path / "merged.yml"

    conda_path.write_text("name: test-env\ndependencies:\n  - numpy")
    pip_path.write_text("requests==2.31.0")

    args = SimpleNamespace(
        conda=conda_path,
        pip=pip_path,
        output=output_path,
        strict=False,
        cpu_only=False,
        log_level="INFO",
        pip_format="txt",  # ★ 明示指定
    )

    run_cli(args, logger)

    assert output_path.exists()
    assert "numpy" in output_path.read_text()


def test_run_cli_with_pip_format_json(tmp_path, logger):
    conda_path = tmp_path / "env.yml"
    pip_path = tmp_path / "req.json"
    output_path = tmp_path / "merged.yml"

    conda_path.write_text("name: test-env\ndependencies:\n  - numpy")
    pip_path.write_text('[{"name": "requests", "version": "2.31.0"}]')

    args = SimpleNamespace(
        conda=conda_path,
        pip=pip_path,
        output=output_path,
        strict=False,
        cpu_only=False,
        log_level="INFO",
        pip_format="json",  # ★ 明示指定
    )

    run_cli(args, logger)

    assert output_path.exists()
    merged = output_path.read_text()
    assert "numpy" in merged
    assert "requests" in merged
