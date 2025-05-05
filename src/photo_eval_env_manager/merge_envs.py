import os
import sys
import yaml
import argparse
from photo_eval_env_manager.envmerge.file_utils import run_pip_list, load_exclude_list
from photo_eval_env_manager.envmerge.pip_utils import parse_pip_input, run_security_audit, parse_pip_package
from photo_eval_env_manager.envmerge.env_utils import normalize_python_version, deduplicate_python, validate_dependencies, validate_versions
from photo_eval_env_manager.envmerge.exceptions import (
    InvalidVersionError,
    DuplicatePackageError,
    VersionMismatchError
)


def merge_envs(base_yml, pip_json, final_yml, requirements_txt, ci_yml=None,
               exclude_for_ci=None, strict=False, dry_run=False, only_pip=False, audit=False):

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

    # pip セクションを dependencies から抽出
    pip_section = []
    for dep in dependencies:
        if isinstance(dep, dict) and 'pip' in dep:
            pip_section.extend(dep['pip'])

    # --- 重複パッケージのチェック（conda側） ---
    conda_names = [parse_pip_package(dep) for dep in dependencies if isinstance(dep, str)]
    conda_dups = {name for name in conda_names if conda_names.count(name) > 1}
    if conda_dups:
        versions = [dep for dep in dependencies if isinstance(dep, str) and parse_pip_package(dep) in conda_dups]
        raise DuplicatePackageError(sorted(conda_dups), versions)

    # --- 重複パッケージのチェック（pip側） ---
    pip_names = [parse_pip_package(pkg) for pkg in pip_section]
    pip_dups = {name for name in pip_names if pip_names.count(name) > 1}
    if pip_dups:
        raise DuplicatePackageError(f"Duplicate pip packages: {sorted(pip_dups)}")

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

    version_mismatches = []

    pip_name_to_line = {parse_pip_package(line): line for line in clean_pip_lines}

    for pkg in clean_pip_lines:
        name = parse_pip_package(pkg)
        if name in conda_pkgs:
            conda_ver = conda_pkgs[name].split('=')[-1] if '=' in conda_pkgs[name] else 'unknown'
            pip_ver = pkg.split('==')[-1]
            if conda_ver != pip_ver:
                print(f"⚠️ Version mismatch: {name} (conda: {conda_ver}, pip: {pip_ver})")
                version_mismatches.append((name, conda_ver, pip_ver))
        if name not in existing_pip:
            pip_section.append(pkg)

    if strict:
        all_names = set(conda_pkgs.keys()).union(pip_name_to_line.keys())
        for name in all_names:
            in_conda = name in conda_pkgs
            in_pip = name in pip_name_to_line

            if in_conda and in_pip:
                continue  # すでに上のループでバージョン不一致はチェック済み
            elif in_conda and not in_pip:
                conda_ver = conda_pkgs[name].split('=')[-1] if '=' in conda_pkgs[name] else 'unknown'
                version_mismatches.append((name, conda_ver, 'missing in pip'))
            elif in_pip and not in_conda:
                pip_ver = pip_name_to_line[name].split('==')[-1]
                version_mismatches.append((name, 'missing in conda', pip_ver))

    if strict and version_mismatches:
        mismatch_msgs = [f"{name}: conda={cv}, pip={pv}" for name, cv, pv in version_mismatches]
        raise VersionMismatchError("Version mismatch detected:\n" + "\n".join(mismatch_msgs))

    pip_section.sort()
    dependencies = deduplicate_python(dependencies)
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
    parser.add_argument('--ci', default=None)
    parser.add_argument('--exclude-for-ci', default=None)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--only-pip', action='store_true')
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()

    try:
        merge_envs(args.base, args.pip_json, args.final, args.requirements,
                   args.ci, args.exclude_for_ci, args.strict, args.dry_run,
                   args.only_pip, args.audit)
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
