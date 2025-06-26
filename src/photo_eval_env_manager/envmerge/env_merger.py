import yaml
import json
import re
import warnings
from typing import Optional
from pathlib import Path
from utils.app_logger import AppLogger
from photo_eval_env_manager.envmerge.exceptions import VersionMismatchError
from constants.env_constants import (
    DEFAULT_PYTHON_VERSION,
    GPU_PACKAGE_REPLACEMENTS,
    is_gpu_package
)

class EnvMerger:
    """
    Conda および pip 依存関係を統合・変換・出力するユーティリティクラス。
    """

    def __init__(
        self,
        base_yml: Path = None,
        pip_json: Path = None,
        cpu_only: bool = False,
        strict: bool = False,
        env_name: str = "photo_eval_env"
    ):
        self.base_yml = base_yml
        self.pip_json = pip_json
        self.cpu_only = cpu_only
        self.strict = strict
        self.env_name = env_name
        self.channels = ["defaults", "conda-forge"]
        self.conda_deps = []
        self.pip_deps = []
        self.logger = AppLogger(logger_name="EnvMerger").get_logger()

    # ------------------------
    # メインAPI（使用順）
    # ------------------------

    def merge_from_sources(
        self,
        conda_file: Path = None,
        pip_file: Path = None,
        cpu_only: bool = False,
        exclude_for_ci: list[str] = None,
        ci_output_path: Path = None,
    ) -> None:
        """
        複数のソースから依存関係を読み込み、正規化・競合解決・CI対応を行う。

        Args:
            conda_file (Path): conda 環境ファイルパス
            pip_file (Path): pip requirements ファイルパス
            cpu_only (bool): GPU パッケージを CPU 用に置換
            exclude_for_ci (list[str]): 除外対象パッケージ
            ci_output_path (Path): CI 用 YAML 出力パス
        """
        if conda_file:
            self.load_conda_file(conda_file)
        if pip_file:
            self.load_pip_file(pip_file)

        self.pip_deps = list(dict.fromkeys(self.pip_deps))  # 重複除去

        self.normalize()
        self.replace_gpu_packages(cpu_only)

        if exclude_for_ci and ci_output_path:
            self.save_ci_yaml(ci_output_path, exclude_for_ci)

    def load(self) -> None:
        """
        コンストラクタで指定されたパスから依存関係を読み込む。
        """
        if self.base_yml:
            self.load_conda_file(self.base_yml)
        if self.pip_json:
            self.load_pip_file(self.pip_json)

    def apply_patches(self) -> None:
        """
        条件に応じて依存関係にパッチ（例: GPU→CPU）を適用する。
        """
        if self.cpu_only:
            self.replace_gpu_packages(cpu_only=True)

    def resolve(self) -> None:
        """
        conda/pip の同名パッケージ間のバージョン矛盾を検出（strict 時）。
        """
        pip_versions = self._parse_pip_versions()
        for conda_pkg in self.conda_deps:
            name, version = self._split_conda_package(conda_pkg)
            if name in pip_versions:
                pip_version = pip_versions[name]
                if self.strict and version and pip_version and version != pip_version:
                    raise VersionMismatchError(f"{name}: conda={version}, pip={pip_version}")
                # ★ 警告: condaにバージョンなし、pipにあり
                if not version and pip_version:
                    warnings.warn(
                        f"[警告] {name} に pip 側でのみバージョン指定 ({pip_version}) が見つかりましたが、conda 側にバージョン指定がありません。",
                        stacklevel=2
                    )

        self._deduplicate_pip_deps()
        self._normalize_conda_versions()    

        # 依存関係を安定化させるためにソート
        self.conda_deps = sorted(self.conda_deps, key=str)
        self.pip_deps = sorted(self.pip_deps)

    def normalize(self) -> None:
        """
        Python バージョンの正規化と依存関係の検証・統合。
        """
        self._normalize_and_deduplicate_python()
        self._validate_dependencies()

        if self.pip_deps:
            self.conda_deps = [d for d in self.conda_deps if not (isinstance(d, dict) and "pip" in d)]
            self.conda_deps.append({"pip": self.pip_deps})

    # ------------------------
    # 入力ローダー
    # ------------------------

    def load_conda_file(self, path: Path) -> None:
        """
        conda YAML ファイルから依存関係を読み込む。
        """
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] Conda file not found: {path}")
        with path.open("r") as f:
            data = yaml.safe_load(f)
        self.conda_deps, self.pip_deps = self._parse_conda_yaml(data)

    def load_pip_file(self, path: Path) -> None:
        """
        pip の JSON または requirements.txt ファイルを読み込む。
        """
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] pip file not found: {path}")
        with path.open("r") as f:
            content = f.read()
        new_deps = self._parse_pip_requirements(content)
        self.pip_deps.extend(new_deps)

    # ------------------------
    # 出力系
    # ------------------------

    def export(self, final_yml: Optional[Path], requirements_txt: Optional[Path]) -> None:
        """
        現在の依存関係を YAML と requirements.txt に書き出す。
        """
        if final_yml is not None:
            with final_yml.open("w") as f:
                yaml.dump(self.build_env_dict(), f, sort_keys=False)

        if requirements_txt is not None:
            with requirements_txt.open("w") as f:
                f.write("\n".join(sorted(self.pip_deps)))

    def write_yaml(self, path: Path) -> None:
        """
        環境 YAML を書き出す。
        """
        with path.open("w") as f:
            yaml.dump(self.build_env_dict(), f, sort_keys=False)

    def write_requirements(self, path: Path) -> None:
        """
        pip requirements.txt を書き出す。
        """
        with path.open("w") as f:
            f.writelines(pkg + "\n" for pkg in self.pip_deps)

    def save_ci_yaml(self, path: Path, exclude: list[str]) -> None:
        """
        除外リストを考慮して CI 用 YAML を書き出す。
        CI 環境名は 'ci-env' に強制変更される。
        """
        filtered = self.filter_for_ci(exclude)
        env_dict = self.build_env_dict(pip_overrides=filtered)
        env_dict["name"] = "ci-env"  # 強制的に上書き
        with path.open("w") as f:
            yaml.dump(env_dict, f, sort_keys=False)

    def filter_for_ci(self, exclude: list[str]) -> list[str]:
        """
        指定パッケージを除外した pip 依存リストを返す。
        """
        exclude_set = {x.lower() for x in exclude}
        return [
            pkg for pkg in self.pip_deps
            if re.split(r"[=<>!~]+", pkg, maxsplit=1)[0].lower() not in exclude_set
        ]

    def build_env_dict(self, pip_overrides: list[str] = None) -> dict:
        """
        現在の依存関係から conda 環境定義（dict）を構築。
        """
        deps = [d for d in self.conda_deps if not (isinstance(d, dict) and "pip" in d)]
        env = {"name": self.env_name, "channels": self.channels, "dependencies": deps}
        pip_section = pip_overrides if pip_overrides is not None else self.pip_deps
        if pip_section:
            env["dependencies"].append({"pip": sorted(pip_section)})
        return env

    def build_env_dict_for_ci(self, exclude_pip_names: list[str]) -> dict:
        """
        CI用の conda 環境定義（dict）を構築。指定された pip パッケージ名を除外。

        Args:
            exclude_pip_names (list[str]): 除外対象のパッケージ名（小文字）
        Returns:
            dict: 除外後の conda 環境定義
        """
        exclude_set = set(exclude_pip_names)

        def get_pkg_name(pkg: str) -> str:
            return re.split(r"[=<>!~]+", pkg, 1)[0].lower()

        filtered_pip = [pkg for pkg in self.pip_deps if get_pkg_name(pkg) not in exclude_set]
        return self.build_env_dict(pip_overrides=filtered_pip)

    # ------------------------
    # 内部処理（private methods）
    # ------------------------

    def _deduplicate_pip_deps(self) -> None:
        """
        pip_depsの重複を排除する。
        strict=Trueの場合はバージョン矛盾があればVersionMismatchErrorを投げる。
        strict=Falseの場合は最初の出現バージョンを優先する。
        """
        unique = {}
        for dep in self.pip_deps:
            pkg_name, sep, ver = dep.partition("==")
            if pkg_name in unique:
                if self.strict and unique[pkg_name] != dep:
                    raise VersionMismatchError(f"Conflicting versions for {pkg_name}: {unique[pkg_name]} vs {dep}")
            else:
                unique[pkg_name] = dep

        self.pip_deps = list(unique.values())
        
    def _parse_conda_yaml(self, data: dict) -> tuple[list[str | dict], list[str]]:
        deps = data.get("dependencies", [])
        conda, pip = [], []
        for d in deps:
            if isinstance(d, dict) and "pip" in d:
                pip.extend(d["pip"])
            else:
                conda.append(d)
        return conda, pip

    def _parse_pip_requirements(self, content: str) -> list[str]:
        try:
            json_data = json.loads(content)
            if isinstance(json_data, list):
                return [f"{pkg['name']}=={pkg['version']}" for pkg in json_data]
        except json.JSONDecodeError:
            pass
        return [line.strip() for line in content.strip().splitlines() if line and not line.startswith("#")]

    def _parse_pip_versions(self) -> dict:
        versions = {}
        for dep in self.pip_deps:
            if "==" in dep:
                name, ver = dep.split("==", 1)
                versions[name.strip()] = ver.strip()
            else:
                versions[dep.strip()] = None
        return versions

    def _split_conda_package(self, dep: str) -> tuple[str, str | None]:
        if "=" in dep:
            name, *ver = dep.split("=")
            return name.strip(), "=".join(ver).strip()
        return dep.strip(), None

    def _normalize_and_deduplicate_python(self) -> None:
        python_indices = [i for i, dep in enumerate(self.conda_deps) if isinstance(dep, str) and dep.lower().startswith("python")]

        if not python_indices:
            self.logger.info("Adding python=3.10 to dependencies")
            self.conda_deps.insert(0, f"python={DEFAULT_PYTHON_VERSION}")
            return

        first_idx = python_indices[0]
        first_dep = self.conda_deps[first_idx]
        ver = first_dep.split("=")[-1] if "=" in first_dep else ""

        if "," in ver or not re.fullmatch(r"3\.10(\.\*)?", ver):
            self.logger.warning(f"Replacing python spec: {first_dep} → python=3.10")
            self.conda_deps[first_idx] = f"python={DEFAULT_PYTHON_VERSION}"

        for idx in reversed(python_indices[1:]):
            self.logger.warning(f"Duplicate python removed: {self.conda_deps[idx]}")
            del self.conda_deps[idx]

    def _normalize_conda_versions(self) -> None:
        """
        特定の conda パッケージのバージョンを正規化。
        例: python=3.9 → python=3.10 に統一
        """
        normalized = []
        for pkg in self.conda_deps:
            name, version = self._split_conda_package(pkg)
            if name == "python" and version == "3.9":
                normalized.append(f"python={DEFAULT_PYTHON_VERSION}")
            else:
                normalized.append(pkg)
        self.conda_deps = normalized

    def _validate_dependencies(self) -> None:
        for dep in self.conda_deps:
            if isinstance(dep, str):
                if dep.lower().startswith("python") and "," in dep:
                    self.logger.error(f"Invalid python version spec: {dep}")
                if not self._validate_version_string(dep):
                    self.logger.warning(f"Unusual version spec: {dep}")
            elif isinstance(dep, dict) and "pip" in dep:
                for pkg in dep["pip"]:
                    if "==" not in pkg:
                        self.logger.warning(f"Unpinned pip package: {pkg}")

    def _validate_version_string(self, line: str) -> bool:
        return bool(re.match(r"^[\w\-]+([=<>!]=?[\w\.\*]+)?$", line))

    def replace_gpu_packages(self, cpu_only: bool) -> None:
        """
        GPU 関連パッケージを CPU 用に置換・削除する。
        """
        if not cpu_only:
            return

        def get_replacement(pkg_name: str) -> str | None:
            return GPU_PACKAGE_REPLACEMENTS.get(pkg_name)

        # pip 置換
        new_pip = []
        for pkg in self.pip_deps:
            name = re.split(r"[=<>!~]+", pkg, 1)[0].lower()
            if is_gpu_package(name):
                repl = get_replacement(name)
                if repl is None:
                    self.logger.info(f"Removing GPU-only pip package: {pkg}")
                    continue
                version = re.sub(r"\+cu\d+", "", pkg[len(name):])
                new_pkg = f"{repl}{version}"
                self.logger.info(f"Replacing {pkg} → {new_pkg}")
                new_pip.append(new_pkg)
            else:
                new_pip.append(pkg)
        self.pip_deps = new_pip

        # conda 置換
        new_conda = []
        for dep in self.conda_deps:
            if isinstance(dep, str):
                name = dep.split("=")[0].lower()
                if is_gpu_package(name):
                    repl = get_replacement(name)
                    if repl is None:
                        self.logger.info(f"Removing GPU-only conda package: {dep}")
                        continue
                    self.logger.info(f"Replacing {dep} → {repl}")
                    new_conda.append(repl)
                else:
                    new_conda.append(dep)
            else:
                new_conda.append(dep)
        self.conda_deps = new_conda
