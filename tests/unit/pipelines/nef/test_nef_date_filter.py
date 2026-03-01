from pathlib import Path

from photo_insight.pipelines.nef.nef_batch_process import NEFFileBatchProcess


def test_nef_find_date_dirs(tmp_path: Path):
    base = tmp_path / "input"
    (base / "2026" / "2026-02-17").mkdir(parents=True)
    (base / "2026" / "20260217").mkdir(parents=True)

    proc = NEFFileBatchProcess(config_path=None, max_workers=1)
    proc.config["base_directory"] = str(base)

    dirs = proc._find_date_dirs(Path(proc.config["base_directory"]), "2026-02-17")
    assert len(dirs) >= 1
    assert any(d.name in ("2026-02-17", "20260217") for d in dirs)
