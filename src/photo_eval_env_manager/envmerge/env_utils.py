import sys
import re
import yaml
import json

from pathlib import Path
from photo_eval_env_manager.envmerge.exceptions import InvalidVersionError

ENV_NAME = "photo_eval_env"


def validate_version_string(pkg_line):
    """
    パッケージ文字列が正しいバージョン指定形式かを検証する。

    :param pkg_line: パッケージ名とバージョンの文字列
    :return: バージョン指定が正しければ True、そうでなければ False
    """
    pattern = re.compile(r"^[a-zA-Z0-9_\-]+([=<>!]=?[0-9a-zA-Z\.\*]+)?$")
    return bool(pattern.match(pkg_line))


def validate_dependencies(dependencies):
    """
    conda の依存関係リストに対してバージョン指定の妥当性を検証する。

    - Python の複数バージョン指定（カンマ区切りなど）がある場合はエラー終了。
    - バージョンが不正な形式であれば警告を表示。
    - pip パッケージでバージョンが指定されていないものがあれば警告。

    :param dependencies: 依存関係リスト（文字列や pip セクションを含む dict）
    """
    for dep in dependencies:
        if isinstance(dep, str):
            if dep.lower().startswith("python") and ',' in dep:
                print(f"❌ Invalid python specifier (multiple versions?): {dep}")
                sys.exit(1)
            if not validate_version_string(dep):
                print(f"⚠️ Invalid version format: {dep}")
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_pkg in dep['pip']:
                if '==' not in pip_pkg:
                    print(f"⚠️ No version specified for pip package: {pip_pkg}")


def normalize_python_version(dependencies):
    """
    Python のバージョン指定が不正または省略されている場合に `python=3.10` に修正・追加する。

    :param dependencies: conda の依存関係リスト（編集対象）
    """
    python_idx = -1
    for i, dep in enumerate(dependencies):
        if isinstance(dep, str) and dep.lower().startswith("python"):
            version_spec = dep.split('=', 1)[-1] if '=' in dep else ''
            if ',' in version_spec or not re.fullmatch(r'3\.10(\.\*)?', version_spec):
                print(f"⚠️ Replacing invalid python spec: {dep} → python=3.10")
                dependencies[i] = "python=3.10"
            python_idx = i
            break
    if python_idx == -1:
        print("✅ Adding python=3.10 to dependencies (was missing)")
        dependencies.insert(0, "python=3.10")


def deduplicate_python(dependencies):
    """
    Python の依存関係が複数含まれていた場合、最初の 1 件を残して他は除去する。

    :param dependencies: 依存関係リスト
    :return: 重複を除去した新しいリスト
    """
    seen = False
    filtered = []
    for dep in dependencies:
        if isinstance(dep, str) and dep.lower().startswith("python"):
            if not seen:
                filtered.append(dep)
                seen = True
            else:
                print(f"⚠️ Removing duplicate python entry: {dep}")
        else:
            filtered.append(dep)
    return filtered


def validate_versions(conda_packages: dict, pip_packages: list):
    """
    conda と pip に同一パッケージが含まれている場合に、バージョンの整合性を検証する。

    :param conda_packages: conda パッケージ名 → バージョン指定文字列の辞書
    :param pip_packages: pip パッケージを表す dict（name, version） のリスト
    :raises InvalidVersionError: 同名パッケージでバージョンが異なる場合に発生
    """
    print("🔥 validate_versions called")
    for pip_pkg in pip_packages:
        name = pip_pkg['name'].lower()
        pip_ver = pip_pkg['version']

        conda_entry = conda_packages.get(name)

        if conda_entry:
            conda_ver = conda_entry.split('=')[-1] if '=' in conda_entry else None
            if conda_ver and conda_ver != pip_ver:
                raise InvalidVersionError(
                    f"Package '{name}' version mismatch: conda='{conda_ver}', pip='{pip_ver}'"
                )


def load_yaml_file(path: Path) -> dict:
    """
    YAML ファイルを読み込んで辞書として返す。

    :param path: 読み込む YAML ファイルのパス
    :return: YAML の内容を格納した辞書
    :raises FileNotFoundError: ファイルが存在しない場合
    """
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Conda environment file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_conda_yaml(data: dict) -> tuple[list, list]:
    """
    conda 環境ファイルのデータから、conda と pip の依存関係を分離する。

    :param data: YAML を読み込んだ辞書形式のデータ
    :return: (conda 依存リスト, pip セクションの依存リスト)
    """
    deps = data.get("dependencies", [])
    conda_deps = []
    pip_section = []
    for dep in deps:
        if isinstance(dep, dict) and "pip" in dep:
            pip_section.extend(dep["pip"])
        else:
            conda_deps.append(dep)
    return conda_deps, pip_section


def parse_pip_requirements(content: str) -> list[str]:
    """
    pip の依存関係（requirements.txt または JSON）を解析してパッケージリストを返す。

    :param content: ファイル内容（テキストまたは JSON）
    :return: パッケージの文字列リスト（"package==version" 形式）
    """
    try:
        pip_data = json.loads(content)
        if isinstance(pip_data, list):
            return [f"{pkg['name']}=={pkg['version']}" for pkg in pip_data]
    except json.JSONDecodeError:
        pass  # 通常のテキスト形式として処理

    lines = content.strip().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def build_merged_env_dict(conda_deps: list, pip_deps: list) -> dict:
    """
    conda と pip の依存関係をマージして 1 つの環境辞書にまとめる。

    :param conda_deps: conda 依存関係リスト
    :param pip_deps: pip 依存関係リスト
    :return: マージされた conda 環境（辞書形式）
    """
    filtered_deps = [dep for dep in conda_deps if not (isinstance(dep, dict) and 'pip' in dep)]

    env = {
        "name": ENV_NAME,
        "channels": ["defaults", "conda-forge"],
        "dependencies": filtered_deps
    }
    if pip_deps:
        env["dependencies"].append({"pip": pip_deps})
    return env
