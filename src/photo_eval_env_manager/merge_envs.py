#!/usr/bin/env python3
import argparse
import yaml
import sys
import re
import json

from pathlib import Path
from photo_eval_env_manager.envmerge.exceptions import VersionMismatchError, DuplicatePackageError
from collections import defaultdict

ENV_NAME = "photo_eval_env"
PYTHON_VERSION = "3.10"


def parse_args():
    parser = argparse.ArgumentParser(description="Merge conda and pip dependencies into a reproducible environment.")
    parser.add_argument("--conda", type=Path, required=True, help="Path to environment.yml")
    parser.add_argument("--pip", type=Path, required=True, help="Path to requirements.txt")
    parser.add_argument("--output", type=Path, default="merged_env.yml", help="Output merged environment file")
    parser.add_argument("--strict", action="store_true", help="Fail if packages overlap between conda and pip")
    parser.add_argument("--cpu-only", action="store_true", help="Replace GPU packages with CPU equivalents (e.g., PyTorch)")
    return parser.parse_args()


def load_conda_env(path):
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Conda environment file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    deps = data.get("dependencies", [])
    conda_deps = []
    pip_section = []
    for dep in deps:
        if isinstance(dep, dict) and "pip" in dep:
            pip_section.extend(dep["pip"])
        else:
            conda_deps.append(dep)
    return conda_deps, pip_section

def load_pip_reqs(path):
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Pip requirements file not found: {path}")
    
    with open(path, "r") as f:
        content = f.read().strip()

    try:
        # JSON array の場合（e.g. [{"name":..., "version":...}, ...]）
        pip_data = json.loads(content)
        if isinstance(pip_data, list):
            return [f"{pkg['name']}=={pkg['version']}" for pkg in pip_data]
    except json.JSONDecodeError:
        pass  # 普通のテキスト形式として扱う

    # 通常の行ベース
    lines = content.splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]

def normalize_pkg_name(pkg):
    if isinstance(pkg, str):
        return re.split(r"[=<>]", pkg.strip())[0].lower()
    return ""


def resolve_conflicts(conda_deps, pip_deps, strict):
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

        # 重複パッケージに関するバージョン情報を収集
        for name in overlap:
            versions_by_name[name].add(conda_pkgs[name])
            versions_by_name[name].add(pip_pkgs[name])

    # 重複バージョンチェック
    for name, versions in versions_by_name.items():
        if len(versions) > 1:
            raise DuplicatePackageError(
                package=name,
                versions=list(versions)
            )


def apply_cpu_patch(pip_deps, conda_deps, cpu_only):
    if not cpu_only:
        return

    # pip の torch バージョンから "+cu111" を削除
    for i, pkg in enumerate(pip_deps):
        if re.match(r"^torch(\W|$)", pkg):
            pip_deps[i] = re.sub(r"\+cu.*", "", pkg)

    # conda の "pytorch" → "pytorch-cpu"
    for i, dep in enumerate(conda_deps):
        if isinstance(dep, str) and dep.startswith("pytorch"):
            conda_deps[i] = dep.replace("pytorch", "pytorch-cpu", 1)

def ensure_python_version(conda_deps):
    if not any(isinstance(pkg, str) and pkg.startswith("python=") for pkg in conda_deps):
        conda_deps.insert(0, f"python={PYTHON_VERSION}")


def write_merged_env(conda_deps, pip_deps, output_path):
    # conda_deps から pip ブロックを除去
    filtered_deps = []
    for dep in conda_deps:
        # dep が dict で、かつ 'pip' キーがある場合は除外
        if isinstance(dep, dict) and 'pip' in dep:
            continue
        filtered_deps.append(dep)

    env = {
        "name": ENV_NAME,
        "channels": ["defaults", "conda-forge"],
        "dependencies": filtered_deps
    }
    if pip_deps:
        env["dependencies"].append({"pip": pip_deps})

    with open(output_path, "w") as f:
        yaml.dump(env, f, sort_keys=False, default_flow_style=False)
    print(f"Merged environment written to: {output_path}")


def merge_envs(base_yml, pip_json, final_yml, requirements_txt,
               ci_yml=None, exclude_for_ci=None, strict=False, cpu_only=False):
    # ✅ ここで全部 Path に変換しておく
    base_yml = Path(base_yml)
    pip_json = Path(pip_json)
    final_yml = Path(final_yml)
    requirements_txt = Path(requirements_txt)
    if ci_yml is not None:
        ci_yml = Path(ci_yml)

    conda_deps, pip_from_conda = load_conda_env(base_yml)
    pip_deps = load_pip_reqs(pip_json)
    pip_deps.extend(pip_from_conda)

    resolve_conflicts(conda_deps, pip_deps, strict)
    apply_cpu_patch(pip_deps, conda_deps, cpu_only)

    write_merged_env(conda_deps, pip_deps, final_yml)
    with requirements_txt.open("w") as f:
        for pkg in pip_deps:
            f.write(pkg + "\n")

    if ci_yml and exclude_for_ci:
        exclude_set = {name.lower() for name in exclude_for_ci}
        def get_pkg_name(dep):
            return dep.split('==')[0].lower()
        ci_pip_deps = [pkg for pkg in pip_deps if get_pkg_name(pkg) not in exclude_set]
        write_merged_env(conda_deps, ci_pip_deps, ci_yml)



def main():
    args = parse_args()

    exclude_for_ci = []
    if args.cpu_only:  # 例: 除外リストを固定ファイルから読んでもよい
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
