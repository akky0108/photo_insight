from __future__ import annotations

from pathlib import Path
import types
import yaml
import csv

import pytest

from photo_insight.pipelines.nef.nef_batch_process import NEFFileBatchProcess


def _write_cfg(tmp_path: Path, base_dir: Path, extra: dict | None = None) -> Path:
    cfg = {
        "base_directory": str(base_dir),
        "output_directory": "temp",  # fallback path (run_ctxが無い場合)
        "append_mode": False,
        # exif_fields はデフォルトでもいいが、テストの期待を固定したいならここで指定
    }
    if extra:
        cfg.update(extra)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


def test_requires_base_directory(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"append_mode": False}), encoding="utf-8")
    with pytest.raises(ValueError):
        NEFFileBatchProcess(config_path=str(cfg))


def test_load_data_collects_nef_case_insensitive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base"
    session = base / "2026-02-01"
    session.mkdir(parents=True)

    # 大文字/小文字の両方を用意
    (session / "a.NEF").write_bytes(b"")
    (session / "b.nef").write_bytes(b"")
    (session / "c.txt").write_text("nope", encoding="utf-8")

    cfg = _write_cfg(tmp_path, base)
    p = NEFFileBatchProcess(config_path=str(cfg))

    # raw_extensions をテストで固定（ExifFileHandlerの実装差に依存しない）
    monkeypatch.setattr(p.exif_handler, "raw_extensions", [".nef"], raising=False)

    # setup() が target_dirs を収集する
    p.setup()
    data = p.load_data()

    paths = sorted([d["path"] for d in data])
    assert len(paths) == 2
    assert paths[0].lower().endswith(".nef")
    assert paths[1].lower().endswith(".nef")


def test_target_dir_sets_subdir_name_to_session_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base"
    session = base / "2026-02-17"
    nested = session / "nested"
    nested.mkdir(parents=True)

    # ネストしてても拾う
    (nested / "x.NEF").write_bytes(b"")

    cfg = _write_cfg(tmp_path, base)
    p = NEFFileBatchProcess(config_path=str(cfg))
    monkeypatch.setattr(p.exif_handler, "raw_extensions", [".nef"], raising=False)

    # target_dir を注入
    p.target_dir = session
    p.setup()
    data = p.load_data()
    assert len(data) == 1
    assert data[0]["subdir_name"] == "2026-02-17"  # session名に固定される


def test_setup_uses_run_ctx_artifacts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base"
    (base / "S1").mkdir(parents=True)

    cfg = _write_cfg(tmp_path, base)
    p = NEFFileBatchProcess(config_path=str(cfg))

    # run_ctx + persist を有効化
    out_dir = tmp_path / "runs_out"
    out_dir.mkdir()
    p.run_ctx = types.SimpleNamespace(out_dir=str(out_dir))
    p._persist_run_results = True

    # target_dir が無い場合 session_name は ALL
    p.setup()
    assert str(p.temp_dir).endswith(str(Path("artifacts") / "nef" / "ALL"))
    assert p.temp_dir.exists()


def test_process_batch_writes_csv_and_skips_when_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "base"
    session = base / "S1"
    session.mkdir(parents=True)
    (session / "a.NEF").write_bytes(b"")

    cfg = _write_cfg(tmp_path, base, extra={"append_mode": False})
    p = NEFFileBatchProcess(config_path=str(cfg))

    # run_ctx を使わず、temp配下に閉じる（テスト用）
    p.project_root = tmp_path  # BaseBatchProcessor が project_root を持つ前提の簡易注入

    # exif読み取りをモック（外部ツール/実ファイル依存を排除）
    def fake_read_file(_path: str) -> dict:
        return {"FileName": Path(_path).name, "ISO": "100", "BitDepth": "14"}

    monkeypatch.setattr(p.exif_handler, "read_file", fake_read_file, raising=True)
    monkeypatch.setattr(p.exif_handler, "raw_extensions", [".nef"], raising=False)

    p.setup()
    data = p.load_data()
    batches = p._generate_batches(data)
    assert len(batches) == 1

    # 1回目：CSVを書く
    p._process_batch(batches[0])
    out = next(p.temp_dir.glob("*_raw_exif_data.csv"))
    assert out.exists()

    # 2回目：append_mode=False & 既存あり → skip（ファイルが増えない）
    before = list(p.temp_dir.glob("*_raw_exif_data.csv"))
    p._process_batch(batches[0])
    after = list(p.temp_dir.glob("*_raw_exif_data.csv"))
    assert len(before) == len(after)
