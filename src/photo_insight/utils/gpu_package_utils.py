# utils/gpu_package_utils.py
from constants.gpu_package_rules import GPU_REPLACEMENTS, GPU_SUFFIXES
import re


def is_gpu_package(pkg_name: str) -> bool:
    name = pkg_name.split("==")[0].lower()
    if name in GPU_REPLACEMENTS:
        return True
    return any(suffix in pkg_name for suffix in GPU_SUFFIXES)


def convert_to_cpu_package(pkg: str) -> str | None:
    name = re.split(r"[=<>!~]+", pkg, 1)[0].lower()
    repl = GPU_REPLACEMENTS.get(name)
    if repl is None:
        return None
    # バージョン文字列 +cu*** を除去
    version = re.sub(r"\+cu\d+", "", pkg[len(name) :])
    return f"{repl}{version}" if repl != name else pkg
