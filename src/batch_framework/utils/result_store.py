# src/batch_framework/utils/result_store.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone


@dataclass(frozen=True)
class RunContext:
    run_id: str
    out_dir: Path
    tmp_dir: Path
    meta: Dict[str, Any] = field(default_factory=dict)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(text: str, dst: Path, encoding: str = "utf-8") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(dst.parent), encoding=encoding
    ) as f:
        f.write(text)
        tmp_path = Path(f.name)
    os.replace(str(tmp_path), str(dst))


def _atomic_write_bytes(data: bytes, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dst.parent)) as f:
        f.write(data)
        tmp_path = Path(f.name)
    os.replace(str(tmp_path), str(dst))


def _atomic_copytree(src: Path, dst: Path) -> None:
    """
    NAS 等へ「完成物」を移す用途。
    dst が既にあれば置換（安全のため dst_tmp→rename）。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dst.parent)) as td:
        tmp = Path(td) / (dst.name + ".tmp")
        shutil.copytree(src, tmp)
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(str(tmp), str(dst))


class ResultStore:
    """
    “正本は Parquet” などにしたいが依存を増やしたくない場合もあるので、
    まずは JSONL/CSV を atomic で書ける形にしておく。
    """

    def __init__(
        self,
        *,
        base_dir: str | Path = "runs",
        use_date_partition: bool = True,
        final_dir: Optional[str | Path] = None,  # 例: NAS mount
    ) -> None:
        self.base_dir = Path(base_dir)
        self.use_date_partition = use_date_partition
        self.final_dir = Path(final_dir) if final_dir is not None else None

    def make_run_context(self, *, prefix: str = "run", ensure_dirs: bool = False) -> RunContext:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}_{stamp}_{os.getpid()}"

        if self.use_date_partition:
            date_dir = datetime.now().strftime("%Y-%m-%d")
            out_dir = self.base_dir / date_dir / run_id
        else:
            out_dir = self.base_dir / run_id

        tmp_dir = out_dir / "_tmp"
        if ensure_dirs:
            tmp_dir.mkdir(parents=True, exist_ok=True)

        meta = {"run_id": run_id, "created_at_utc": _now_utc_iso()}
        return RunContext(run_id=run_id, out_dir=out_dir, tmp_dir=tmp_dir, meta=meta)

    def save_meta(self, ctx: RunContext, extra: Optional[Dict[str, Any]] = None) -> None:
        meta = dict(ctx.meta)
        if extra:
            meta.update(extra)
        _atomic_write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            ctx.out_dir / "meta.json",
        )

    def save_jsonl(self, ctx: RunContext, *, rows: list[dict], name: str = "results.jsonl") -> Path:
        # JSONL は schema 揺れがあっても落ちにくい
        lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
        path = ctx.out_dir / name
        _atomic_write_text(lines, path)
        return path

    def finalize_to_final_dir(self, ctx: RunContext) -> Optional[Path]:
        if self.final_dir is None:
            return None

        # base_dir からの相対を保って final にコピー（date partition を維持）
        try:
            rel = ctx.out_dir.relative_to(self.base_dir)
        except Exception:
            # 念のため：relative が取れない場合は run_id 配下へ
            rel = Path(ctx.out_dir.name)

        dst = self.final_dir / rel
        _atomic_copytree(ctx.out_dir, dst)
        return dst
