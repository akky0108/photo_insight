import pytest
from pathlib import Path
from photo_eval_env_manager.envmerge.env_merger import EnvMerger
from photo_eval_env_manager.envmerge.exceptions import VersionMismatchError, DuplicatePackageError


# === 基本マージ処理のテスト ===

def test_merge_from_sources_basic(tmp_path):
    """Conda + Pip の両ファイルから環境をマージする基本ケース"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text("""
name: test-env
dependencies:
  - numpy=1.21.0
  - python=3.9
  - pip
  - pip:
      - requests==2.31.0
""")

    req_txt.write_text("""
flask==2.1.0
requests==2.31.0
""")

    merger = EnvMerger()
    merger.merge_from_sources(conda_file=env_yml, pip_file=req_txt)

    # python=3.9 → 3.10 に補正されている想定
    assert "python=3.10" in merger.conda_deps
    assert "numpy=1.21.0" in merger.conda_deps
    assert "flask==2.1.0" in merger.pip_deps
    assert merger.pip_deps.count("requests==2.31.0") == 1  # 重複は除去されている


def test_merge_from_sources_cpu_only(tmp_path):
    """--cpu-only オプションが有効な場合のマージ挙動を検証"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text("""
name: test-env
dependencies:
  - pytorch
  - pip
""")

    req_txt.write_text("""
torch==2.0.0+cu118
tensorflow==2.15.0
""")

    merger = EnvMerger()
    merger.merge_from_sources(conda_file=env_yml, pip_file=req_txt, cpu_only=True)

    assert "pytorch-cpu" in merger.conda_deps
    assert "torch-cpu==2.0.0" in merger.pip_deps
    assert "tensorflow-cpu==2.15.0" in merger.pip_deps
    assert all("cudatoolkit" not in dep for dep in merger.conda_deps)


def test_merge_from_sources_ci_output(tmp_path):
    """CI用出力に指定されたパッケージが除外されるか検証"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"
    ci_yml = tmp_path / "ci_env.yml"

    env_yml.write_text("""
name: test-env
dependencies:
  - numpy
  - pip
  - pip:
      - requests==2.31.0
      - flask==2.1.0
""")

    req_txt.write_text("pytest==7.2.0\nflask==2.1.0")

    merger = EnvMerger()
    merger.merge_from_sources(
        conda_file=env_yml,
        pip_file=req_txt,
        exclude_for_ci=["flask", "pytest"],
        ci_output_path=ci_yml
    )

    content = ci_yml.read_text()
    assert ci_yml.exists()
    assert "flask" not in content
    assert "pytest" not in content
    assert "requests==2.31.0" in content
    assert "- pip:" in content


# === 単体メソッドのテスト ===

def test_load(tmp_path):
    """load() によって conda/pip の依存関係が読み込まれるか検証"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text("""
name: test-env
dependencies:
  - numpy=1.21.0
  - python=3.10
  - pip
  - pip:
      - requests==2.31.0
""")

    req_txt.write_text("""
flask==2.1.0
requests==2.31.0
""")

    merger = EnvMerger(base_yml=env_yml, pip_json=req_txt)
    merger.load()

    assert "numpy=1.21.0" in merger.conda_deps
    assert "flask==2.1.0" in merger.pip_deps
    assert "requests==2.31.0" in merger.pip_deps


def test_resolve_strict_conflict(tmp_path):
    """strict=True 時にバージョン不一致があれば VersionMismatchError を出す"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text("""
name: test-env
dependencies:
  - requests==2.30.0
  - pip
  - pip:
      - flask
""")

    req_txt.write_text("requests==2.31.0")

    merger = EnvMerger(base_yml=env_yml, pip_json=req_txt, strict=True)
    merger.load()

    with pytest.raises(VersionMismatchError):
        merger.resolve()


def test_apply_cpu_patch():
    """apply_patches() で CUDA付きのパッケージ名を CPU版へ変換できるか検証"""
    merger = EnvMerger(base_yml=None, pip_json=None, cpu_only=True)
    merger.pip_deps = ["torch==2.0.0+cu118", "flask==2.1.0"]
    merger.conda_deps = ["pytorch", "numpy"]

    merger.apply_patches()

    assert "torch-cpu==2.0.0" in merger.pip_deps
    assert merger.pip_deps[0] == "torch-cpu==2.0.0"
    assert merger.conda_deps[0] == "pytorch-cpu"


def test_export(tmp_path):
    """export() によって conda/pip の依存ファイルが正しく出力されるか検証"""
    env_yml = tmp_path / "environment.yml"
    req_txt = tmp_path / "requirements.txt"

    env_yml.write_text("""
name: test-env
dependencies:
  - numpy
  - python=3.10
  - pip
""")

    req_txt.write_text("flask==2.1.0")

    merger = EnvMerger(base_yml=env_yml, pip_json=req_txt)
    merger.load()
    merger.resolve()
    merger.apply_patches()

    final_env = tmp_path / "final.yml"
    final_req = tmp_path / "final_requirements.txt"
    merger.export(final_yml=final_env, requirements_txt=final_req)

    assert final_env.exists()
    assert final_req.exists()
    assert "flask==2.1.0" in final_req.read_text()
    assert "numpy" in final_env.read_text()
    assert "python=3.10" in final_env.read_text()
