import os
import sys
import yaml
import argparse
from typing import List, Dict
from photo_eval_env_manager.envmerge.file_utils import run_pip_list, load_exclude_list
from photo_eval_env_manager.envmerge.pip_utils import parse_pip_input, run_security_audit, parse_pip_package
from photo_eval_env_manager.envmerge.env_utils import normalize_python_version, deduplicate_python, validate_dependencies
from photo_eval_env_manager.envmerge.exceptions import (
    InvalidVersionError,
    DuplicatePackageError,
    VersionMismatchError
)

# 重複チェックのための効率的な関数
def check_pip_duplicates(pip_section: List[str]):
    pip_names = [parse_pip_package(pkg) for pkg in pip_section]
    pip_dups = {name for name in pip_names if pip_names.count(name) > 1}
    if pip_dups:
        raise DuplicatePackageError(f"Duplicate pip packages: {sorted(pip_dups)}")

# バージョン不一致チェックを効率化
def check_version_mismatches(conda_pkgs: Dict[str, str], pip_lines: List[str], strict: bool):
    version_mismatches = []
    pip_name_to_line = {parse_pip_package(line): line for line in pip_lines}

    for line in pip_lines:
        name = parse_pip_package(line)
        if name in conda_pkgs:
            conda_ver = conda_pkgs[name].split('=')[-1] if '=' in conda_pkgs[name] else 'unknown'
            pip_ver = line.split('==')[-1]
            if conda_ver != pip_ver:
                version_mismatches.append((name, conda_ver, pip_ver))

    if strict:
        # 両方に存在しないパッケージをチェック
        for name in set(conda_pkgs).union(pip_name_to_line):
            if name not in conda_pkgs:
                version_mismatches.append((name, 'missing in conda', pip_name_to_line[name].split('==')[-1]))
            elif name not in pip_name_to_line:
                version_mismatches.append((name, conda_pkgs[name], 'missing in pip'))

    if version_mismatches:
        msg = '\n'.join([f"{name}: conda={cv}, pip={pv}" for name, cv, pv in version_mismatches])
        raise VersionMismatchError(f"Version mismatch detected:\n{msg}")

# CPU専用バージョンへの変換関数
def to_cpu_version(pkg: str) -> str:
    if pkg.startswith("torch==") and "+cu" in pkg:
        return pkg.split("+")[0]
    return pkg

# requirements.txt の書き込み処理
def write_requirements_txt(pip_section, requirements_txt, dry_run):
    if dry_run:
        print("\n🧺 [dry-run] requirements.txt:")
        print('\n'.join(pip_section))
    else:
        with open(requirements_txt, 'w') as f:
            f.write('\n'.join(pip_section))
        print(f"✅ Created {requirements_txt}")

def merge_envs(base_yml, pip_json, final_yml, requirements_txt, ci_yml=None,
               exclude_for_ci=None, strict=False, dry_run=False, only_pip=False,
               audit=False, cpu_only=False):

    if not os.path.exists(base_yml):
        raise FileNotFoundError(f"Base YAML not found: {base_yml}")
    if not os.path.exists(pip_json):
        raise FileNotFoundError(f"pip list JSON not found: {pip_json}")

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
        if cpu_only:
            pip_section = [to_cpu_version(pkg) for pkg in clean_pip_lines]
        else:
            pip_section = clean_pip_lines
        write_requirements_txt(pip_section, requirements_txt, dry_run)

        if audit:
            run_security_audit(requirements_txt)
        return

    with open(base_yml, 'r') as f:
        base_env = yaml.safe_load(f)

    dependencies = base_env.get('dependencies', [])
    normalize_python_version(dependencies)
    validate_dependencies(dependencies)

    pip_section = []
    conda_names = []
    conda_pkgs = {}

    for dep in dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_section.extend(dep['pip'])
        elif isinstance(dep, str):
            name = parse_pip_package(dep)
            conda_names.append(name)
            conda_pkgs[name] = dep

    # 重複チェック
    conda_dups = {name for name in conda_names if conda_names.count(name) > 1}
    if conda_dups:
        versions = [dep for dep in dependencies if isinstance(dep, str) and parse_pip_package(dep) in conda_dups]
        raise DuplicatePackageError(sorted(conda_dups), versions)

    check_pip_duplicates(pip_section)
    check_version_mismatches(conda_pkgs, clean_pip_lines, strict)

    # pip セクション更新
    pip_section = sorted(set(pip_section + clean_pip_lines))
    dependencies = deduplicate_python(dependencies)

    # 書き込み
    base_env['dependencies'] = [
        dep for dep in dependencies if not (isinstance(dep, dict) and 'pip' in dep)
    ] + [{'pip': pip_section}]

    if dry_run:
        print("\n🧺 [dry-run] Final environment_combined.yml:")
        print(yaml.dump(base_env, sort_keys=False))
        write_requirements_txt(pip_section, requirements_txt, dry_run)
    else:
        with open(final_yml, 'w') as f:
            yaml.dump(base_env, f, sort_keys=False)
        print(f"✅ Created {final_yml}")

        write_requirements_txt(pip_section, requirements_txt, dry_run)

    if audit:
        run_security_audit(requirements_txt)

    if ci_yml and exclude_for_ci:
        if isinstance(exclude_for_ci, str):
            exclude_for_ci = load_exclude_list(exclude_for_ci)

        base_env_ci = base_env.copy()
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
            if cpu_only:
                for dep in base_env_ci['dependencies']:
                    if isinstance(dep, dict) and 'pip' in dep:
                        dep['pip'] = [to_cpu_version(pkg) for pkg in dep['pip']]
            with open(ci_yml, 'w') as f:
                yaml.dump(base_env_ci, f, sort_keys=False)
            print(f"✅ Created {ci_yml}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conda Environment Merge Tool")
    parser.add_argument('--base', default='environment_base.yml')
    parser.add_argument('--pip-json', default='pip_list.json')
    parser.add_argument('--final', default='environment_combined.yml')
    parser.add_argument('--requirements', default='requirements.txt')
    parser.add_argument('--ci', default=None)
    parser.add_argument('--exclude-for-ci', default=None)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--only-pip', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()

    try:
        merge_envs(
            base_yml=args.base,
            pip_json=args.pip_json,
            final_yml=args.final,
            requirements_txt=args.requirements,
            ci_yml=args.ci,
            exclude_for_ci=args.exclude_for_ci,
            strict=args.strict,
            dry_run=args.dry_run,
            only_pip=args.only_pip,
            audit=args.audit,
            cpu_only=args.cpu_only
        )
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except InvalidVersionError as e:
        print(f"❌ Invalid version format or mismatch: {e}")
        sys.exit(1)
    except DuplicatePackageError as e:
        print(f"❌ Duplicate package found: {e}")
        sys.exit(1)
    except VersionMismatchError as e:
        print(f"❌ Version mismatch error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
