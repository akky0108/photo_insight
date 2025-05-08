#!/usr/bin/env python3
import argparse
import yaml
import sys
from pathlib import Path

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
    with open(path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def resolve_conflicts(conda_deps, pip_deps, strict):
    if not strict:
        return
    conda_pkgs = {pkg.split("=")[0] if isinstance(pkg, str) else "" for pkg in conda_deps}
    pip_pkgs = {pkg.split("==")[0] for pkg in pip_deps}
    overlap = conda_pkgs & pip_pkgs
    if overlap:
        raise RuntimeError(f"Conflicting packages found in conda and pip: {overlap}")


def apply_cpu_patch(pip_deps, conda_deps, cpu_only):
    if not cpu_only:
        return
    pip_deps[:] = [pkg for pkg in pip_deps if not pkg.startswith("torch")]
    conda_deps[:] = [pkg if pkg != "pytorch" else "pytorch-cpu" for pkg in conda_deps]


def write_merged_env(conda_deps, pip_deps, output_path):
    env = {
        "name": ENV_NAME,
        "channels": ["defaults", "conda-forge"],
        "dependencies": conda_deps
    }
    if pip_deps:
        env["dependencies"].append({"pip": pip_deps})

    with open(output_path, "w") as f:
        yaml.dump(env, f, default_flow_style=False)
    print(f"Merged environment written to: {output_path}")


def main():
    args = parse_args()

    conda_deps, pip_from_conda = load_conda_env(args.conda)
    pip_deps = load_pip_reqs(args.pip)
    pip_deps.extend(pip_from_conda)

    resolve_conflicts(conda_deps, pip_deps, args.strict)
    apply_cpu_patch(pip_deps, conda_deps, args.cpu_only)

    write_merged_env(conda_deps, pip_deps, args.output)


if __name__ == "__main__":
    main()
