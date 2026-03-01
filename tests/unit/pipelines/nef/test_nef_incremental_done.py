from pathlib import Path
from unittest.mock import MagicMock

from photo_insight.pipelines.nef.nef_batch_process import NEFFileBatchProcess


def test_nef_incremental_skips_done_marker(tmp_path: Path):
    out = tmp_path / "output"
    out.mkdir(parents=True)

    proc = NEFFileBatchProcess(config_path=None, max_workers=1)
    proc.config["paths"] = {"output_data_dir": str(out)}
    proc.temp_dir = tmp_path / "runs" / "dummy"
    proc.temp_dir.mkdir(parents=True, exist_ok=True)

    proc.nef_incremental = True
    proc.exif_handler.read_file = MagicMock(return_value={"FileName": "x.NEF"})

    nef_path = tmp_path / "input" / "x.NEF"
    nef_path.parent.mkdir(parents=True, exist_ok=True)
    nef_path.write_bytes(b"dummy")

    abs_p = str(nef_path.resolve())
    marker = proc._done_marker(abs_p)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("done\n", encoding="utf-8")

    batch = [{"path": str(nef_path), "directory": str(nef_path.parent), "filename": nef_path.name, "subdir_name": "S1"}]

    proc._process_batch(batch)

    proc.exif_handler.read_file.assert_not_called()
