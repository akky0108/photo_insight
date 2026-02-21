import pytest
import yaml
import warnings
from photo_insight.photo_eval_env_manager.envmerge.env_merger import EnvMerger
from photo_insight.photo_eval_env_manager.envmerge.exceptions import (
    VersionMismatchError,
)

# === 基本マージ処理のテスト ===


def test_merge_from_sources_basic(tmp_path):
    """Conda + Pip の両ファイルから環境をマージする基本ケース"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text(
        """
name: test-env
dependencies:
  - numpy=1.21.0
  - python=3.9
  - pip
  - pip:
      - requests==2.31.0
"""
    )

    req_txt.write_text(
        """
flask==2.1.0
requests==2.31.0
"""
    )

    merger = EnvMerger()
    merger.merge_from_sources(conda_file=env_yml, pip_file=req_txt)

    # python=3.9 → 3.10 に補正されている想定
    assert "python=3.10" in merger.conda_deps
    assert "numpy=1.21.0" in merger.conda_deps
    assert "flask==2.1.0" in merger.pip_deps
    assert merger.pip_deps.count("requests==2.31.0") == 1  # 重複は除去されている


def test_python_version_upgraded_from_3_9_to_3_10(tmp_path):
    """python=3.9 が python=3.10 に正規化されるか検証"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text(
        """
name: test-env
dependencies:
  - python=3.9
  - numpy
  - pip
"""
    )
    req_txt.write_text("")

    merger = EnvMerger(base_yml=env_yml, pip_json=req_txt)
    merger.load()
    merger.resolve()

    assert "python=3.10" in merger.conda_deps
    assert "python=3.9" not in merger.conda_deps


def test_pip_duplicate_deduplication(tmp_path):
    """pipパッケージが複数ソースに存在しても重複が除去されることを検証"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text(
        """
name: test-env
dependencies:
  - pip
  - pip:
      - requests==2.31.0
"""
    )

    req_txt.write_text(
        """
requests==2.31.0
flask==2.1.0
"""
    )

    merger = EnvMerger()
    merger.merge_from_sources(conda_file=env_yml, pip_file=req_txt)

    assert merger.pip_deps.count("requests==2.31.0") == 1
    assert "flask==2.1.0" in merger.pip_deps


def test_export_sorted_pip_deps(tmp_path):
    """requirements.txt に出力される pip 依存がソート済みであることを検証"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text(
        """
name: test-env
dependencies:
  - pip
"""
    )

    req_txt.write_text(
        """
flask==2.1.0
requests==2.31.0
numpy==1.24.0
"""
    )

    merger = EnvMerger(base_yml=env_yml, pip_json=req_txt)
    merger.load()
    merger.resolve()
    merger.apply_patches()

    out_req = tmp_path / "out_requirements.txt"
    merger.export(final_yml=None, requirements_txt=out_req)

    lines = [line.strip() for line in out_req.read_text().splitlines() if line.strip()]
    assert lines == sorted(lines)


def test_resolve_prefers_first_version_when_strict_false():
    """strict=False 時は最初のバージョンが優先される"""
    merger = EnvMerger(base_yml=None, pip_json=None, strict=False)
    merger.pip_deps = ["flask==2.0.0", "flask==2.1.0", "requests==2.31.0"]

    merger.resolve()

    assert "flask==2.0.0" in merger.pip_deps
    assert "flask==2.1.0" not in merger.pip_deps
    assert merger.pip_deps.count("flask==2.0.0") == 1


def test_deduplicate_pip_deps_non_strict():
    """strict=False の場合、最初のバージョンを優先して pip の重複が除去される"""
    merger = EnvMerger(strict=False)
    merger.pip_deps = [
        "requests==2.30.0",
        "flask==2.1.0",
        "requests==2.30.0",
    ]

    merger._deduplicate_pip_deps()

    assert merger.pip_deps.count("requests==2.30.0") == 1
    assert "flask==2.1.0" in merger.pip_deps


def test_deduplicate_pip_deps_strict_conflict():
    """strict=True の場合、pip に同一パッケージ名で異なるバージョンが存在するとエラー"""
    merger = EnvMerger(strict=True)
    merger.pip_deps = [
        "requests==2.30.0",
        "requests==2.31.0",
    ]

    with pytest.raises(VersionMismatchError) as e:
        merger._deduplicate_pip_deps()

    assert "requests" in str(e.value)


def test_resolve_warns_on_missing_conda_version(tmp_path):
    """conda 側にバージョンがなく pip 側にだけあるときに警告が出るか検証"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text(
        """
name: test-env
dependencies:
  - requests
  - pip
"""
    )
    req_txt.write_text("requests==2.31.0")

    merger = EnvMerger(base_yml=env_yml, pip_json=req_txt, strict=False)
    merger.load()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        merger.resolve()
        warning_msgs = [str(warning.message) for warning in w]

    assert any("requests に pip 側でのみバージョン指定" in msg for msg in warning_msgs)


def test_save_ci_yaml_excludes_and_sorts(tmp_path):
    """save_ci_yaml() が除外リストを反映し、pip セクションをソートして出力するかを確認"""
    env_yml = tmp_path / "environment.yml"
    ci_yml = tmp_path / "ci_output.yml"

    env_yml.write_text(
        """
name: test-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - flask==2.2.0
      - requests==2.31.0
      - numpy==1.23.0
"""
    )

    merger = EnvMerger(base_yml=env_yml)
    merger.load()
    merger.save_ci_yaml(path=ci_yml, exclude=["flask"])

    assert ci_yml.exists()
    content = ci_yml.read_text()
    assert "flask" not in content
    assert "requests==2.31.0" in content
    assert "numpy==1.23.0" in content

    ci_dict = yaml.safe_load(content)
    pip_deps = next(
        d["pip"] for d in ci_dict["dependencies"] if isinstance(d, dict) and "pip" in d
    )
    assert pip_deps == sorted(pip_deps)


def test_save_ci_yaml_sets_env_name(tmp_path):
    """CI 出力で env_name が 'ci-env' に強制されることを確認"""
    env_yml = tmp_path / "environment.yml"
    ci_yml = tmp_path / "ci_output.yml"

    env_yml.write_text(
        """
name: original-env
dependencies:
  - python=3.10
  - pip
"""
    )

    merger = EnvMerger(base_yml=env_yml)
    merger.load()
    merger.save_ci_yaml(path=ci_yml, exclude=[])

    content = yaml.safe_load(ci_yml.read_text())
    assert content["name"] == "ci-env"


def test_sorted_deps_after_resolve(tmp_path):
    """resolve() 後に conda/pip の依存関係が昇順ソートされることを確認"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text(
        """
name: test-env
dependencies:
  - pip
  - zlib
  - numpy
  - pip:
      - requests==2.31.0
      - flask==2.1.0
"""
    )
    req_txt.write_text("flask==2.1.0\npytest==7.2.0")

    merger = EnvMerger(base_yml=env_yml, pip_json=req_txt)
    merger.load()
    merger.resolve()

    assert merger.conda_deps == sorted(merger.conda_deps, key=str)
    assert merger.pip_deps == sorted(merger.pip_deps)


def test_filter_for_ci_with_various_versions():
    """除外対象にバージョン付き・比較演算子付きがあっても正しく除去できるか"""
    merger = EnvMerger()
    merger.pip_deps = [
        "Flask==2.1.0",
        "requests>=2.30.0",
        "numpy!=1.24.0",
        "PyTest~=7.2",
    ]
    filtered = merger.filter_for_ci(exclude=["flask", "pytest"])

    assert "Flask==2.1.0" not in filtered
    assert "PyTest~=7.2" not in filtered
    assert "requests>=2.30.0" in filtered
    assert "numpy!=1.24.0" in filtered


def test_replace_gpu_packages_cpu_only():
    """GPU パッケージが CPU 対応に置換されるか確認"""
    merger = EnvMerger(cpu_only=True)
    merger.conda_deps = ["pytorch"]
    merger.pip_deps = ["torch==2.1.0+cu118"]
    merger.replace_gpu_packages(cpu_only=True)

    assert merger.conda_deps == ["pytorch-cpu"]
    assert merger.pip_deps == ["torch-cpu==2.1.0"]


def test_filter_for_ci_case_insensitive():
    """除外リストが大文字小文字を区別せず適用されることを確認"""
    merger = EnvMerger()
    merger.pip_deps = ["Flask==2.1.0", "pytest==7.2.0"]
    filtered = merger.filter_for_ci(exclude=["FLASK", "PyTest"])

    assert filtered == []
