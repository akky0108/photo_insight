import yaml
import json
import re
import warnings
from typing import Optional
from pathlib import Path
from photo_insight.utils.app_logger import AppLogger
from photo_insight.photo_eval_env_manager.envmerge.exceptions import (
    VersionMismatchError,
    DuplicatePackageError,
)
from photo_insight.constants.env_constants import (
    DEFAULT_PYTHON_VERSION,
    GPU_PACKAGE_REPLACEMENTS,
    is_gpu_package,
)
from collections import defaultdict


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
        env_name: str = "photo_eval_env",
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
        conda_strs = [d for d in self.conda_deps if isinstance(d, str)]
        self._check_duplicate_packages(conda_strs, source="conda", allow_exact_duplicates=not self.strict)

        if self.strict:
            self._check_duplicate_packages(self.pip_deps, source="pip", allow_exact_duplicates=False)
            # strict=False なら重複チェックはしない（後段のdedupで先勝ちに寄せる）
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
                        stacklevel=2,
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

    def load_pip_file(self, path: Path, format: Optional[str] = None) -> None:
        """
        pip の JSON または requirements.txt ファイルを読み込む。

        Args:
            path (Path): ファイルパス
            format (str, optional): 'json' または 'txt' を明示指定。None の場合は自動判別。

        Raises:
            FileNotFoundError: ファイルが存在しない
            ValueError: パースに失敗した場合、または形式が不明な場合
        """
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] pip file not found: {path}")

        content = path.read_text(encoding="utf-8")

        try:
            if format == "json":
                self.logger.info(f"Loading pip dependencies as JSON: {path}")
                deps = self._parse_pip_json(content)

            elif format == "txt":
                self.logger.info(f"Loading pip dependencies as plain text: {path}")
                deps = self._parse_pip_txt(content)

            elif format is None:
                # 自動判別：まず JSON として読み込み、失敗したら TXT
                self.logger.info(f"Auto-detecting format for pip file: {path}")
                try:
                    deps = self._parse_pip_json(content)
                    self.logger.info(f"Auto-detected JSON format: {path}")
                except json.JSONDecodeError as json_err:
                    self.logger.warning(f"Failed to parse as JSON: {json_err}")
                    try:
                        deps = self._parse_pip_txt(content)
                        self.logger.info(f"Falling back to TXT format: {path}")
                    except Exception as txt_err:
                        raise ValueError(
                            f"[ERROR] Failed to parse pip file '{path}' as JSON or TXT.\n"
                            f"- JSON error: {json_err}\n"
                            f"- TXT error: {txt_err}\n"
                            f"→ Please specify format explicitly using format='json' or 'txt'."
                        )
            else:
                raise ValueError(f"[ERROR] Unknown format '{format}'. Use 'json', 'txt', or None for auto-detect.")

        except Exception:
            self.logger.exception(f"Failed to load pip file: {path}")
            raise

        self.pip_deps.extend(deps)

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
        return [pkg for pkg in self.pip_deps if re.split(r"[=<>!~]+", pkg, maxsplit=1)[0].lower() not in exclude_set]

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

    def _parse_pip_json(self, content: str) -> list[str]:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list of {name, version} objects")
        return [
            f"{pkg['name']}=={pkg['version']}"
            for pkg in data
            if isinstance(pkg, dict) and "name" in pkg and "version" in pkg
        ]

    def _parse_pip_txt(self, content: str) -> list[str]:
        return [
            line.strip() for line in content.strip().splitlines() if line.strip() and not line.strip().startswith("#")
        ]

    def _split_conda_package(self, dep: str) -> tuple[str, str | None]:
        if "=" in dep:
            name, *ver = dep.split("=")
            return name.strip(), "=".join(ver).strip()
        return dep.strip(), None

    def _normalize_and_deduplicate_python(self) -> None:
        python_indices = [
            i for i, dep in enumerate(self.conda_deps) if isinstance(dep, str) and dep.lower().startswith("python")
        ]

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
            name = re.split(r"[=<>!~]+", pkg, 1)[0].strip().lower()
            if is_gpu_package(name):
                repl = get_replacement(name)
                if repl is None:
                    self.logger.info(f"Removing GPU-only pip package: {pkg}")
                    continue

                # e.g. "torch==2.1.0+cu118" -> op="==", ver="2.1.0+cu118"
                m = re.search(r"(==|>=|<=|>|<|~=)\s*([^\s;]+)", pkg)
                if m:
                    op, ver = m.group(1), m.group(2)
                    # drop local version suffix like "+cu118"
                    ver = ver.split("+", 1)[0]
                    version = f"{op}{ver}"
                else:
                    version = ""

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

    def _check_duplicate_packages(self, packages: list[str], source: str, allow_exact_duplicates: bool = False) -> None:
        """
        パッケージリスト内の重複パッケージ名（バージョン含む）を検出し、エラーを投げる。
        """

        seen = defaultdict(list)
        for pkg in packages:
            name = re.split(r"[=<>!~]+", pkg)[0].strip().lower()
            seen[name].append(pkg.strip())

        for name, versions in seen.items():
            unique_lowered = set(v.lower() for v in versions)
            if len(versions) > 1:
                if allow_exact_duplicates and len(unique_lowered) == 1:
                    continue  # 全て同一表記ならOK
                raise DuplicatePackageError(name, versions)
