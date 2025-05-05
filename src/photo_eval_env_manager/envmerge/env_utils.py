import sys
import re
from photo_eval_env_manager.envmerge.exceptions import InvalidVersionError

def validate_version_string(pkg_line):
    pattern = re.compile(r"^[a-zA-Z0-9_\-]+([=<>!]=?[0-9a-zA-Z\.\*]+)?$")
    return bool(pattern.match(pkg_line))

def validate_dependencies(dependencies):
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
    """Remove invalid or compound python version specs and replace with python=3.10"""
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
    バージョン不一致を検証し、もし不一致があれば InvalidVersionError を発生させる。

    :param conda_packages: conda 環境のパッケージとそのバージョンを持つ辞書
    :param pip_packages: pip パッケージのリスト
    :raises InvalidVersionError: バージョンが一致しない場合
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
