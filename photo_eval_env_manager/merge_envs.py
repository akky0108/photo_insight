# merge_envs.py
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
    for sep in ['==', '>=', '<=']:
        if sep in line:
            return line.split(sep)[0].strip().lower()
    return line.strip().lower()

def run_pip_list(output_file):
    try:
        print("\n📦 Running 'pip list'...")
        result = subprocess.run(["pip", "list", "--format=json"], check=True, capture_output=True, text=True)
        with open(output_file, 'w') as f:
            f.write(result.stdout)
        print(f"✅ Generated {output_file}")
    except subprocess.CalledProcessError:
        print("❌ Failed to run pip list.")
        sys.exit(1)

def run_security_audit(requirements_txt):
    try:
        print("\n🔍 Running security audit with pip-audit...")
        subprocess.run(["pip-audit", "-r", requirements_txt], check=True)
    except FileNotFoundError:
        print("⚠️ pip-audit not found. Try 'pip install pip-audit'.")
    except subprocess.CalledProcessError:
        print("❌ Security vulnerabilities found.")

def load_exclude_list(file_path):
    try:
        with open(file_path, 'r') as f:
            return set(pkg.strip().lower() for pkg in f if pkg.strip())
    except Exception as e:
        print(f"❌ Failed to read exclude list: {e}")
        return set()

def parse_pip_input(pip_json_path):
    with open(pip_json_path, 'r') as f:
        content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            parsed = []
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                name = parse_pip_package(line)
                version = line.split('==')[-1] if '==' in line else 'unknown'
                parsed.append({"name": name, "version": version})
            return parsed

def validate_version_string(pkg_line):
    pattern = re.compile(r"^[a-zA-Z0-9_\-]+([=<>!]=?[0-9a-zA-Z\.\*]+)?$")
    return bool(pattern.match(pkg_line))

def validate_dependencies(dependencies):
    for dep in dependencies:
        if isinstance(dep, str):
            if not validate_version_string(dep):
                print(f"⚠️ Invalid version format: {dep}")
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_pkg in dep['pip']:
                if '==' not in pip_pkg:
                    print(f"⚠️ No version specified for pip package: {pip_pkg}")

def normalize_python_version(dependencies):
    for i, dep in enumerate(dependencies):
        if isinstance(dep, str) and dep.lower().startswith("python"):
            parts = dep.split('=')
            if len(parts) >= 2 and parts[1].startswith('3.1'):
                print(f"⚠️ Fixing malformed python version: {dep} → python=3.10")
                dependencies[i] = 'python=3.10'

def validate_final_env(env_yml):
    print(f"\n🧪 Validating {env_yml} with conda dry-run...")
    result = subprocess.run(["conda", "env", "create", "-f", env_yml, "--dry-run"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ {env_yml} is invalid:\n{result.stderr}")
    else:
        print(f"✅ {env_yml} is valid.")

def merge_envs(base_yml, pip_json, final_yml, requirements_txt, ci_yml=None, exclude_for_ci=None, strict=False, dry_run=False, only_pip=False, audit=False):
    if not os.path.exists(base_yml):
        print(f"❌ Base YAML not found: {base_yml}")
        sys.exit(1)
    if not os.path.exists(pip_json):
        print(f"❌ pip list JSON not found: {pip_json}")
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
            print("\n🧺 [dry-run] requirements.txt (only-pip):")
            print('\n'.join(clean_pip_lines))
        else:
            with open(requirements_txt, 'w') as f:
                f.write('\n'.join(clean_pip_lines))
            print(f"✅ Created {requirements_txt} (only-pip)")
        if audit:
            run_security_audit(requirements_txt)
        return

    with open(base_yml, 'r') as f:
        base_env = yaml.safe_load(f)

    dependencies = base_env.get('dependencies', [])
    normalize_python_version(dependencies)
    validate_dependencies(dependencies)

    pip_section = None
    for dep in dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_section = dep['pip']
            break

    if pip_section is None:
        pip_section = []
        dependencies.append({'pip': pip_section})

    existing_pip = {parse_pip_package(pkg): pkg for pkg in pip_section}
    conda_pkgs = {parse_pip_package(dep): dep for dep in dependencies if isinstance(dep, str)}

    for pkg in clean_pip_lines:
        name = parse_pip_package(pkg)
        if name in conda_pkgs:
            conda_ver = conda_pkgs[name].split('=')[-1] if '=' in conda_pkgs[name] else 'unknown'
            pip_ver = pkg.split('==')[-1]
            if conda_ver != pip_ver:
                print(f"⚠️ Version mismatch: {name} (conda: {conda_ver}, pip: {pip_ver})")
                if strict:
                    print("❌ Stopped due to strict mode.")
                    sys.exit(1)
        if name not in existing_pip:
            pip_section.append(pkg)

    pip_section.sort()
    base_env['dependencies'] = dependencies

    if dry_run:
        print("\n🧺 [dry-run] Final environment_combined.yml:")
        print(yaml.dump(base_env, sort_keys=False))
        print("\n🧺 [dry-run] requirements.txt:")
        print('\n'.join(pip_section))
    else:
        with open(final_yml, 'w') as f:
            yaml.dump(base_env, f, sort_keys=False)
        print(f"✅ Created {final_yml}")

        with open(requirements_txt, 'w') as f:
            f.write('\n'.join(pip_section))
        print(f"✅ Created {requirements_txt}")

    if audit:
        run_security_audit(requirements_txt)

    if ci_yml and exclude_for_ci:
        base_env_ci = yaml.safe_load(open(final_yml, 'r')) if not dry_run else base_env.copy()
        ci_dependencies = []
        for dep in base_env_ci['dependencies']:
            if isinstance(dep, dict) and 'pip' in dep:
                filtered = [pkg for pkg in dep['pip'] if parse_pip_package(pkg) not in exclude_for_ci]
                if filtered:
                    ci_dependencies.append({'pip': filtered})
            elif isinstance(dep, str):
                ci_dependencies.append(dep)
        base_env_ci['dependencies'] = ci_dependencies

        if dry_run:
            print("\n🧺 [dry-run] environment_ci.yml:")
            print(yaml.dump(base_env_ci, sort_keys=False))
        else:
            with open(ci_yml, 'w') as f:
                yaml.dump(base_env_ci, f, sort_keys=False)
            print(f"✅ Created {ci_yml}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conda Environment Merge Tool")
    parser.add_argument('--base', default='environment_base.yml')
    parser.add_argument('--pip-json', default='pip_list.json')
    parser.add_argument('--final', default='environment_combined.yml')
    parser.add_argument('--requirements', default='requirements.txt')
    parser.add_argument('--ci', default='environment_ci.yml')
    parser.add_argument('--exclude', default='exclude_ci.txt')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--only-pip', action='store_true')
    parser.add_argument('--audit', action='store_true')
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
