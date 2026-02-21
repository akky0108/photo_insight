#!/usr/bin/env python3
import argparse
import yaml
import sys
import logging
import re

from pathlib import Path
from photo_insight.photo_eval_env_manager.envmerge.env_utils import (
    load_yaml_file,
    parse_conda_yaml,
    parse_pip_requirements,
)
from photo_insight.photo_eval_env_manager.envmerge.exceptions import (
    VersionMismatchError,
    DuplicatePackageError,
)
from photo_insight.photo_eval_env_manager.envmerge.env_merger import EnvMerger
from collections import defaultdict
from photo_insight.utils.app_logger import AppLogger


def parse_args():
    """
    コマンドライン引数をパースして返す。

    Returns:
        argparse.Namespace: パースされた引数オブジェクト。
    """
    parser = argparse.ArgumentParser(
        description="Merge conda and pip dependencies into a reproducible environment."
    )
    parser.add_argument(
        "--conda", type=Path, required=True, help="Path to environment.yml"
    )
    parser.add_argument(
        "--pip", type=Path, required=True, help="Path to requirements.txt"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="merged_env.yml",
        help="Output merged environment file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if packages overlap between conda and pip",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Replace GPU packages with CPU equivalents (e.g., PyTorch)",
    )
    parser.add_argument(
        "--LogLevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="ログの詳細レベル（デフォルト: INFO）",
    )
    parser.add_argument(
        "--pip-format",
        choices=["json", "txt"],
        help="Format of the pip file: 'json' or 'txt'. If omitted, auto-detect.",
    )
    return parser.parse_args()


def load_pip_reqs(path: Path) -> list[str]:
    """
    pip用requirementsファイルを読み込んで、依存パッケージのリストを返す。

    Args:
        path (Path): requirements.txtファイルのパス

    Returns:
        list[str]: パッケージのリスト

    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Pip requirements file not found: {path}")

    with open(path, "r") as f:
        content = f.read()

    return parse_pip_requirements(content)


def normalize_pkg_name(pkg):
    """
    パッケージ名からバージョン指定や条件を取り除き、正規化した名前を返す。

    Args:
        pkg (str): パッケージの指定（例: "requests==2.31.0"）

    Returns:
        str: 正規化されたパッケージ名（例: "requests"）
    """
    if isinstance(pkg, str):
        return re.split(r"[=<>]", pkg.strip())[0].lower()
    return ""


def write_merged_env_file(env_dict: dict, output_path: Path):
    """
    環境辞書をYAML形式でファイルに書き出す。

    Args:
        env_dict (dict): conda環境の定義（dependenciesなどを含む）
        output_path (Path): 出力先ファイルパス
    """
    with open(output_path, "w") as f:
        yaml.dump(env_dict, f, sort_keys=False, default_flow_style=False)
    print(f"Merged environment written to: {output_path}")


def merge_envs(
    base_yml,
    pip_json,
    final_yml,
    requirements_txt,
    ci_yml=None,
    exclude_for_ci=None,
    strict=False,
    cpu_only=False,
    logger=None,
    pip_format=None,
):
    """
    condaとpipの依存ファイルをマージして、環境YAMLおよびrequirements.txtを生成する。

    Args:
        base_yml (str or Path): baseとなるconda環境YAML
        pip_json (str or Path): pipパッケージ情報のrequirements.txt
        final_yml (str or Path): 出力する統合conda YAML
        requirements_txt (str or Path): 出力するrequirements.txt
        ci_yml (str or Path, optional): CI用に除外処理したYAMLを出力するパス
        exclude_for_ci (list[str], optional): CI用に除外するパッケージ名リスト
        strict (bool): バージョン重複を厳密にチェックするか
        cpu_only (bool): CPU専用モードを有効にするか
        logger (AppLogger, optional): ログ出力用のロガーインスタンス
    """
    if logger is None:
        logger = AppLogger(project_root=".", logger_name="envmerge").get_logger()

    try:
        base_yml = Path(base_yml)
        pip_json = Path(pip_json)
        final_yml = Path(final_yml)
        requirements_txt = Path(requirements_txt)
        if ci_yml is not None:
            ci_yml = Path(ci_yml)

        if logger:
            logger.debug(f"Loading base conda environment from {base_yml}")
        raw_yaml = load_yaml_file(base_yml)

        if logger:
            logger.debug("Parsing conda dependencies")
        conda_deps, pip_from_conda = parse_conda_yaml(raw_yaml)

        if logger:
            logger.debug(f"Loading pip requirements from {pip_json}")
        pip_deps = load_pip_reqs(pip_json)
        pip_deps.extend(pip_from_conda)

        merger = EnvMerger(cpu_only=cpu_only, strict=strict)
        merger.conda_deps = conda_deps

        merger.load_pip_file(pip_json, format=pip_format)
        merger.pip_deps.extend(pip_from_conda)

        # conflictチェック（今回）
        merger.resolve()

        # GPU → CPU置換（前回）
        merger.replace_gpu_packages(cpu_only=cpu_only)

        # ★ ここで normalize を呼び、python=3.10 を自動追加
        merger.normalize()

        # 書き戻し
        conda_deps = merger.conda_deps
        pip_deps = merger.pip_deps

        # conda 環境定義（dict）を構築。
        env_dict = merger.build_env_dict()
        write_merged_env_file(env_dict=env_dict, output_path=final_yml)

        if logger:
            logger.debug(f"Writing pip requirements to {requirements_txt}")
        with requirements_txt.open("w") as f:
            for pkg in pip_deps:
                f.write(pkg + "\n")

        if ci_yml and exclude_for_ci:
            logger.debug(f"Excluding packages for CI: {exclude_for_ci}")
            ci_env_dict = merger.build_env_dict_for_ci(exclude_for_ci)
            write_merged_env_file(env_dict=ci_env_dict, output_path=ci_yml)

    except Exception:
        if logger:
            logger.exception("An error occurred during merge_envs execution.")
        raise


def run_cli(args, logger):
    """
    CLI 引数と logger を使って merge_envs を実行する。

    Args:
        args (argparse.Namespace): コマンドライン引数
        logger (logging.Logger): ロガーインスタンス
    """
    exclude_for_ci = []
    if args.cpu_only:
        exclude_file = Path("exclude_ci.txt")
        if exclude_file.exists():
            with open(exclude_file) as f:
                exclude_for_ci = [line.strip() for line in f if line.strip()]

    merge_envs(
        base_yml=args.conda,
        pip_json=args.pip,
        final_yml=args.output,
        requirements_txt="requirements.txt",
        ci_yml="environment_ci.yml",
        exclude_for_ci=exclude_for_ci,
        strict=args.strict,
        cpu_only=args.cpu_only,
        logger=logger,
    )


def main():
    """
    スクリプトのエントリーポイント。コマンドライン引数を受け取り、環境マージ処理を実行する。
    """

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--log-level", default="INFO")
    pre_args, _ = pre_parser.parse_known_args()
    log_level = pre_args.log_level.upper()

    logger = AppLogger(project_root=".", logger_name="envmerge").get_logger()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    try:
        args = parse_args()
        logger.setLevel(getattr(logging, args.log_level, logging.INFO))
        run_cli(args, logger)
    except Exception:
        logger.exception("UnHandled exception occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
