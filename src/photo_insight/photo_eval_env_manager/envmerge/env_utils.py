import sys
import re
import yaml
import json

from pathlib import Path
from photo_insight.photo_eval_env_manager.envmerge.exceptions import InvalidVersionError

ENV_NAME = "photo_eval_env"


def validate_version_string(pkg_line: str) -> bool:
    """
    パッケージのバージョン指定が有効な形式かどうかを検証する。

    例: "numpy==1.23.4", "pandas>=1.5" は OK。形式が不正な場合は False を返す。

    :param pkg_line: パッケージ名とバージョン指定を含む文字列
    :return: 正しい形式であれば True、そうでなければ False
    """
    pattern = re.compile(r"^[a-zA-Z0-9_\-]+([=<>!]=?[0-9a-zA-Z\.\*]+)?$")
    return bool(pattern.match(pkg_line))


def validate_dependencies(dependencies: list[str | dict]) -> None:
    """
    conda 環境の依存関係リストをチェックして、バージョン指定が妥当か検証する。

    - Python の複数バージョン指定（例: "python=3.10,>=3.9"）はエラーとして処理。
    - 曖昧または不正な形式のバージョン指定は警告を表示。
    - pip セクション内のパッケージでバージョンが指定されていない場合も警告。

    :param dependencies: conda 環境ファイルの dependencies セクション（str または dict を含むリスト）
    """
    for dep in dependencies:
        if isinstance(dep, str):
            # python の複数指定を検出
            if dep.lower().startswith("python") and "," in dep:
                print(f"❌ Invalid python specifier (multiple versions?): {dep}")
                sys.exit(1)

            # 不正なバージョン形式の検出
            if not validate_version_string(dep):
                print(f"⚠️ Invalid version format: {dep}")

        elif isinstance(dep, dict) and "pip" in dep:
            for pip_pkg in dep["pip"]:
                # pip パッケージにバージョン指定がない場合
                if "==" not in pip_pkg:
                    print(f"⚠️ No version specified for pip package: {pip_pkg}")


def normalize_python_version(dependencies: list[str | dict]) -> None:
    """
    Python のバージョン指定が無効または存在しない場合、"python=3.10" を追加または置換する。

    - バージョン指定がカンマ区切りだったり、"3.10" 以外の場合は警告を出して置換。
    - Python の指定がなければ先頭に追加する。

    :param dependencies: 編集対象の conda 依存関係リスト（インプレースで変更される）
    """
    python_idx = -1
    for i, dep in enumerate(dependencies):
        if isinstance(dep, str) and dep.lower().startswith("python"):
            version_spec = dep.split("=", 1)[-1] if "=" in dep else ""
            if "," in version_spec or not re.fullmatch(r"3\.10(\.\*)?", version_spec):
                print(f"⚠️ Replacing invalid python spec: {dep} → python=3.10")
                dependencies[i] = "python=3.10"
            python_idx = i
            break

    if python_idx == -1:
        print("✅ Adding python=3.10 to dependencies (was missing)")
        dependencies.insert(0, "python=3.10")


def deduplicate_python(dependencies: list[str | dict]) -> list[str | dict]:
    """
    "python" の指定が複数ある場合、最初の 1 件を残して残りを除去する。

    :param dependencies: conda の依存関係リスト
    :return: Python の重複を除いた新しい依存関係リスト
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


def validate_versions(
    conda_packages: dict[str, str], pip_packages: list[dict[str, str]]
) -> None:
    """
    conda と pip の両方に同じパッケージがある場合、バージョンが一致するかを検証する。

    :param conda_packages: パッケージ名をキーとした conda のパッケージ辞書（例: {"numpy": "numpy=1.23.4"}）
    :param pip_packages: "name" と "version" を持つ pip パッケージのリスト（例: [{"name": "numpy", "version": "1.23.4"}]）
    :raises InvalidVersionError: 同一パッケージで conda と pip のバージョンが異なる場合
    """
    print("🔥 validate_versions called")
    for pip_pkg in pip_packages:
        name = pip_pkg["name"].lower()
        pip_ver = pip_pkg["version"]
        conda_entry = conda_packages.get(name)

        if conda_entry:
            conda_ver = conda_entry.split("=")[-1] if "=" in conda_entry else None
            if conda_ver and conda_ver != pip_ver:
                raise InvalidVersionError(
                    f"Package '{name}' version mismatch: conda='{conda_ver}', pip='{pip_ver}'"
                )


def load_yaml_file(path: Path) -> dict:
    """
    指定された YAML ファイルを読み込み、辞書形式で返す。

    ファイルが存在しない場合は FileNotFoundError を送出する。

    :param path: 読み込む YAML ファイルのパス（Path オブジェクト）
    :return: YAML の内容を格納した dict
    :raises FileNotFoundError: 指定されたファイルが存在しない場合
    """
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Conda environment file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_conda_yaml(data: dict) -> tuple[list[str | dict], list[str]]:
    """
    YAML データから conda パッケージと pip パッケージをそれぞれ抽出する。

    pip の依存関係は `{"pip": [...]}` という辞書形式で与えられていることを想定。

    :param data: YAML を読み込んだ dict（environment.yml の内容）
    :return: タプル (conda パッケージのリスト, pip パッケージのリスト)
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
    pip の依存関係（requirements.txt 形式 または JSON 形式）を解析し、パッケージ文字列のリストを返す。

    - JSON の場合は [{"name": ..., "version": ...}] を想定。
    - 通常のテキスト形式では "package==version" の行を抽出。
    - コメントや空行は無視される。

    :param content: requirements ファイルの内容（JSON または テキスト）
    :return: "package==version" 形式のパッケージ名リスト
    """
    try:
        pip_data = json.loads(content)
        if isinstance(pip_data, list):
            return [f"{pkg['name']}=={pkg['version']}" for pkg in pip_data]
    except json.JSONDecodeError:
        pass  # 通常のテキスト形式として処理

    lines = content.strip().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
