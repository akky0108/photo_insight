import json
import subprocess
import re
from photo_insight.photo_eval_env_manager.envmerge.exceptions import InvalidVersionError


def validate_version_string(version: str) -> None:
    """バージョン文字列が正しい形式かどうか検証する。"""
    if not re.match(r"^\d+(\.\d+)*$", version):
        raise InvalidVersionError(version)


def parse_pip_package(line):
    """パッケージ名を行から抽出（lowercaseで返す）"""
    if " @ " in line:
        return line.split(" @ ")[0].strip().lower()
    for sep in ["==", ">=", "<="]:
        if sep in line:
            return line.split(sep)[0].strip().lower()
    return line.strip().lower()


def parse_pip_input(pip_json_path):
    """JSONまたは通常のrequirements.txtからパッケージとバージョンをパース"""
    with open(pip_json_path, "r") as f:
        content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            parsed = []
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name = parse_pip_package(line)
                version = line.split("==")[-1] if "==" in line else "unknown"
                if version != "unknown":
                    validate_version_string(version)
                parsed.append({"name": name, "version": version})
            return parsed


def run_security_audit(requirements_txt):
    """pip-auditによるセキュリティ監査を実行"""
    try:
        print("\n🔍 Running security audit with pip-audit...")
        subprocess.run(["pip-audit", "-r", requirements_txt], check=True)
    except FileNotFoundError:
        print("⚠️ pip-audit not found. Try 'pip install pip-audit'.")
    except subprocess.CalledProcessError:
        print("❌ Security vulnerabilities found.")
