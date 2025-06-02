#!/usr/bin/env python3
import argparse
import yaml
import sys
import re
import json

from pathlib import Path
from photo_eval_env_manager.envmerge.env_utils import load_yaml_file, parse_conda_yaml, parse_pip_requirements, build_merged_env_dict
from photo_eval_env_manager.envmerge.exceptions import VersionMismatchError, DuplicatePackageError
from collections import defaultdict

PYTHON_VERSION = "3.10"


def parse_args():
    """
    コマンドライン引数をパースして返す。

    Returns:
        argparse.Namespace: パースされた引数オブジェクト。
    """
    parser = argparse.ArgumentParser(description="Merge conda and pip dependencies into a reproducible environment.")
    parser.add_argument("--conda", type=Path, required=True, help="Path to environment.yml")
    parser.add_argument("--pip", type=Path, required=True, help="Path to requirements.txt")
    parser.add_argument("--output", type=Path, default="merged_env.yml", help="Output merged environment file")
    parser.add_argument("--strict", action="store_true", help="Fail if packages overlap between conda and pip")
    parser.add_argument("--cpu-only", action="store_true", help="Replace GPU packages with CPU equivalents (e.g., PyTorch)")
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


def resolve_conflicts(conda_deps, pip_deps, strict):
    """
    condaとpipの依存関係の重複・バージョン違いを検出し、必要に応じて例外を投げる。

    Args:
        conda_deps (list[str]): conda依存リスト
        pip_deps (list[str]): pip依存リスト
        strict (bool): 厳密なバージョン一致を求めるか

    Raises:
        VersionMismatchError: 厳密チェック時に重複パッケージが存在する場合
        DuplicatePackageError: 同一パッケージ名でバージョンが異なる場合
    """
    conda_pkgs = {normalize_pkg_name(pkg): pkg for pkg in conda_deps}
    pip_pkgs = {normalize_pkg_name(pkg): pkg for pkg in pip_deps}
    overlap = set(conda_pkgs) & set(pip_pkgs)

    versions_by_name = defaultdict(set)

    if overlap:
        if strict:
            raise VersionMismatchError(
                f"Version mismatch for packages present in both conda and pip: {', '.join(overlap)}"
            )
        else:
            print(f"[warn] Overlapping packages between conda and pip: {', '.join(overlap)}")

        for name in overlap:
            versions_by_name[name].add(conda_pkgs[name])
            versions_by_name[name].add(pip_pkgs[name])

    for name, versions in versions_by_name.items():
        if len(versions) > 1:
            raise DuplicatePackageError(
                package=name,
                versions=list(versions)
            )


def apply_cpu_patch(pip_deps, conda_deps, cpu_only):
    """
    CPU専用環境に変換するため、GPU依存のパッケージ表記を修正する。

    Args:
        pip_deps (list[str]): pip依存リスト
        conda_deps (list[str]): conda依存リスト
        cpu_only (bool): CPU専用モードが有効かどうか
    """
    if not cpu_only:
        return

    for i, pkg in enumerate(pip_deps):
        if re.match(r"^torch(\W|$)", pkg):
            pip_deps[i] = re.sub(r"\+cu.*", "", pkg)

    for i, dep in enumerate(conda_deps):
        if isinstance(dep, str) and dep.startswith("pytorch"):
            conda_deps[i] = dep.replace("pytorch", "pytorch-cpu", 1)


def ensure_python_version(conda_deps):
    """
    conda依存に明示的なPythonバージョンが含まれていない場合、デフォルトを追加する。

    Args:
        conda_deps (list[str]): conda依存リスト（変更可能）
    """
    if not any(isinstance(pkg, str) and pkg.startswith("python=") for pkg in conda_deps):
        conda_deps.insert(0, f"python={PYTHON_VERSION}")


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


def merge_envs(base_yml, pip_json, final_yml, requirements_txt,
               ci_yml=None, exclude_for_ci=None, strict=False, cpu_only=False):
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
    """
    base_yml = Path(base_yml)
    pip_json = Path(pip_json)
    final_yml = Path(final_yml)
    requirements_txt = Path(requirements_txt)
    if ci_yml is not None:
        ci_yml = Path(ci_yml)

    raw_yaml = load_yaml_file(base_yml)
    conda_deps, pip_from_conda = parse_conda_yaml(raw_yaml)
    pip_deps = load_pip_reqs(pip_json)
    pip_deps.extend(pip_from_conda)

    resolve_conflicts(conda_deps, pip_deps, strict)
    apply_cpu_patch(pip_deps, conda_deps, cpu_only)

    write_merged_env_file(build_merged_env_dict(conda_deps=conda_deps, pip_deps=pip_deps), final_yml)
    with requirements_txt.open("w") as f:
        for pkg in pip_deps:
            f.write(pkg + "\n")

    if ci_yml and exclude_for_ci:
        exclude_set = {name.lower() for name in exclude_for_ci}

        def get_pkg_name(dep):
            return dep.split('==')[0].lower()

        ci_pip_deps = [pkg for pkg in pip_deps if get_pkg_name(pkg) not in exclude_set]

        write_merged_env_file(
            build_merged_env_dict(conda_deps=conda_deps, pip_deps=ci_pip_deps),
            ci_yml
        )


def main():
    """
    スクリプトのエントリーポイント。コマンドライン引数を受け取り、環境マージ処理を実行する。
    """
    args = parse_args()

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
        cpu_only=args.cpu_only
    )


if __name__ == "__main__":
    main()
