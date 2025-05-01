import os
import sys
import subprocess
import yaml
import json
import argparse
import re

def parse_pip_package(line):
    if ' @ ' in line:
        return line.split(' @ ')[0].strip().lower()
    if '==' in line:
        return line.split('==')[0].strip().lower()
    if '>=' in line:
        return line.split('>=')[0].strip().lower()
    if '<=' in line:
        return line.split('<=')[0].strip().lower()
    return line.strip().lower()

def run_pip_list(output_file):
    try:
        print("📦 pip list を実行中...")
        result = subprocess.run(["pip", "list", "--format=json"], check=True, capture_output=True, text=True)
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        print(f"✅ {output_file} を生成しました。")
    except subprocess.CalledProcessError:
        print("❌ pip list に失敗しました。")
        sys.exit(1)

def run_security_audit(requirements_txt):
    try:
        print("🔍 セキュリティ監査を実行中 (pip-audit)...")
        subprocess.run(["pip-audit", "-r", requirements_txt], check=True)
    except FileNotFoundError:
        print("⚠️ pip-audit が見つかりません。`pip install pip-audit` を実行してください。")
    except subprocess.CalledProcessError:
        print("❌ セキュリティ監査で脆弱性が見つかりました。レポートを確認してください。")

def load_exclude_list(file_path):
    try:
        with open(file_path, 'r') as f:
            return set(pkg.strip().lower() for pkg in f if pkg.strip())
    except Exception as e:
        print(f"❌ 除外ファイルの読み込みに失敗しました: {e}")
        return set()

def parse_pip_input(pip_json_path):
    with open(pip_json_path, 'r') as f:
        content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # fallback for pip freeze format
            lines = content.splitlines()
            parsed = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                name = parse_pip_package(line)
                version = line.split('==')[-1] if '==' in line else 'unknown'
                parsed.append({"name": name, "version": version})
            return parsed

def validate_version_string(pkg_line):
    """
    Conda のパッケージバージョン指定が不正でないかを確認。
    例: python=3.10 など。
    """
    version_pattern = re.compile(r"^[a-zA-Z0-9_\-]+([=<>!]=?[0-9a-zA-Z\.\*]+)?$")
    return bool(version_pattern.match(pkg_line))

def validate_dependencies(dependencies):
    for dep in dependencies:
        if isinstance(dep, str):
            if not validate_version_string(dep):
                print(f"⚠️ 無効なバージョン指定: {dep}")
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_pkg in dep['pip']:
                if '==' not in pip_pkg:
                    print(f"⚠️ pip パッケージのバージョンが明示されていません: {pip_pkg}")

def merge_envs(base_yml, pip_json, final_yml, requirements_txt, ci_yml=None, exclude_for_ci=None, strict=False, dry_run=False, only_pip=False, audit=False):
    if not os.path.exists(base_yml):
        print(f"❌ ベース環境ファイルが存在しません: {base_yml}")
        sys.exit(1)
    if not os.path.exists(pip_json):
        print(f"❌ pip list のJSONファイルが存在しません: {pip_json}")
        sys.exit(1)

    pip_packages = parse_pip_input(pip_json)

    pip_package_names = set()
    clean_pip_lines = []

    for pkg in pip_packages:
        name = pkg['name'].lower()
        version = pkg['version']
        if name not in pip_package_names:
            pip_package_names.add(name)
            clean_pip_lines.append(f"{name}=={version}")

    clean_pip_lines.sort()

    if only_pip:
        if dry_run:
            print("🧺 [dry-run] requirements.txt の内容 (only-pip):")
            for line in clean_pip_lines:
                print(line)
        else:
            with open(requirements_txt, 'w') as f:
                for line in clean_pip_lines:
                    f.write(f"{line}\n")
            print(f"✅ requirements.txt ({requirements_txt}) を作成しました (only-pip)。")
        if audit:
            run_security_audit(requirements_txt)
        return

    with open(base_yml, 'r') as f:
        base_env = yaml.safe_load(f)

    dependencies = base_env.get('dependencies', [])
    validate_dependencies(dependencies)
    
    pip_section = None
    for dep in dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_section = dep['pip']
            break

    if pip_section is None:
        pip_section = []
        dependencies.append({'pip': pip_section})

    existing_pip_packages = {parse_pip_package(pkg): pkg for pkg in pip_section}
    conda_packages = {
        parse_pip_package(dep): dep
        for dep in dependencies if isinstance(dep, str)
    }

    for pkg in clean_pip_lines:
        pkg_name = parse_pip_package(pkg)
        if pkg_name in conda_packages:
            conda_line = conda_packages[pkg_name]
            conda_version = conda_line.split('=')[-1] if '=' in conda_line else '不明'
            pip_version = pkg.split('==')[-1]
            if conda_version != pip_version:
                msg = f"⚠️ バージョン不一致: {pkg_name} (conda: {conda_version}, pip: {pip_version})"
                print(msg)
                if strict:
                    print("❌ strictモードで停止します。")
                    sys.exit(1)

        if pkg_name not in existing_pip_packages:
            pip_section.append(pkg)

    pip_section.sort()
    base_env['dependencies'] = dependencies

    if dry_run:
        print("🧺 [dry-run] 結合環境ファイルの内容:")
        print(yaml.dump(base_env, sort_keys=False))
        print("🧺 [dry-run] requirements.txt の内容:")
        for line in pip_section:
            print(line)
    else:
        with open(final_yml, 'w') as f:
            yaml.dump(base_env, f, sort_keys=False)
        print(f"✅ 結合環境ファイル {final_yml} を作成しました。")

        with open(requirements_txt, 'w') as f:
            for line in pip_section:
                f.write(f"{line}\n")
        print(f"✅ requirements.txt ({requirements_txt}) を作成しました。")

    if audit:
        run_security_audit(requirements_txt)

    if ci_yml and exclude_for_ci:
        base_env_ci = yaml.safe_load(open(final_yml, 'r')) if not dry_run else base_env.copy()
        ci_dependencies = []
        for dep in base_env_ci['dependencies']:
            if isinstance(dep, dict) and 'pip' in dep:
                filtered_pip = [pkg for pkg in dep['pip'] if parse_pip_package(pkg) not in exclude_for_ci]
                if filtered_pip:
                    ci_dependencies.append({'pip': filtered_pip})
            elif isinstance(dep, str):
                ci_dependencies.append(dep)
        base_env_ci['dependencies'] = ci_dependencies

        if dry_run:
            print("🧺 [dry-run] CI用環境ファイルの内容:")
            print(yaml.dump(base_env_ci, sort_keys=False))
        else:
            with open(ci_yml, 'w') as f:
                yaml.dump(base_env_ci, f, sort_keys=False)
            print(f"✅ CI用環境ファイル {ci_yml} を作成しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conda 環境マージツール")
    parser.add_argument('--base', default='environment_base.yml', help='ベースのenvironment.ymlファイル')
    parser.add_argument('--pip-json', default='pip_list.json', help='pip list結果のJSONファイル')
    parser.add_argument('--final', default='environment_combined.yml', help='出力する最終environment.yml')
    parser.add_argument('--requirements', default='requirements.txt', help='出力するrequirements.txt')
    parser.add_argument('--ci', default='environment_ci.yml', help='CI用の軽量environment.yml')
    parser.add_argument('--exclude', default='exclude_ci.txt', help='CI除外パッケージのファイル')
    parser.add_argument('--strict', action='store_true', help='バージョン衝突があればエラーとして停止')
    parser.add_argument('--dry-run', action='store_true', help='ファイルを書き込まずに差分を表示')
    parser.add_argument('--only-pip', action='store_true', help='requirements.txt のみを出力し、他をスキップ')
    parser.add_argument('--audit', action='store_true', help='requirements.txt に対して pip-audit によるセキュリティ監査を行う')
    args = parser.parse_args()

    script_dir = os.getcwd()

    base_yml = os.path.join(script_dir, args.base)
    pip_json = os.path.join(script_dir, args.pip_json)
    final_yml = os.path.join(script_dir, args.final)
    requirements_txt = os.path.join(script_dir, args.requirements)
    ci_yml = os.path.join(script_dir, args.ci)
    exclude_txt = os.path.join(script_dir, args.exclude)

    run_pip_list(pip_json)
    exclude_for_ci = load_exclude_list(exclude_txt) if not args.only_pip else set()

    merge_envs(
        base_yml, pip_json, final_yml, requirements_txt,
        ci_yml, exclude_for_ci,
        strict=args.strict, dry_run=args.dry_run, only_pip=args.only_pip, audit=args.audit
    )
