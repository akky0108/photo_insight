# src/batch_framework/utils/result_store.py
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunContext:
    """
    1回の実行(run)を表すコンテキスト。

    - out_dir: 出力先ディレクトリ（通常 runs/YYYY-MM-DD/run_xxx）
    - tmp_dir: out_dir 配下の作業用（必要なら使う）
    - meta:    実行メタ情報（meta.json に書く想定）
    """

    run_id: str
    out_dir: Path
    tmp_dir: Path
    meta: Dict[str, Any] = field(default_factory=dict)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(text: str, dst: Path, encoding: str = "utf-8") -> None:
    """
    dst を atomic に書き換える（同一FS上で os.replace）。
    dst.parent は必要に応じて作成する。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(dst.parent), encoding=encoding
    ) as f:
        f.write(text)
        tmp_path = Path(f.name)
    os.replace(str(tmp_path), str(dst))


def _atomic_write_bytes(data: bytes, dst: Path) -> None:
    """
    dst を atomic に書き換える（bytes版）。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dst.parent)) as f:
        f.write(data)
        tmp_path = Path(f.name)
    os.replace(str(tmp_path), str(dst))


def _atomic_copytree(src: Path, dst: Path) -> None:
    """
    ディレクトリ tree を atomic に置換コピーする。

    - dst が存在しても安全に置換する（dst_tmp を作ってから rename）。
    - 最終的に dst は完成品のみになる。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 同一親配下に tmp を作って、完成後に os.replace
    with tempfile.TemporaryDirectory(dir=str(dst.parent)) as td:
        tmp = Path(td) / (dst.name + ".tmp")
        shutil.copytree(src, tmp)

        # 既存があれば削除して置換（Windows/一部FSで os.replace がディレクトリに弱い場合があるため）
        if dst.exists():
            shutil.rmtree(dst)

        os.replace(str(tmp), str(dst))


class ResultStore:
    """
    実行(run)ごとの成果物を保存する軽量ストア。

    設計方針（重要）:
    - make_run_context() はデフォルトでディレクトリを作らない（副作用ゼロ）
    - save_* が呼ばれた時点で必要な親ディレクトリが作られる（atomic write 内）
    - finalize_to_final_dir() は final_dir 指定時のみ動く
    """

    def __init__(
        self,
        *,
        base_dir: str | Path = "runs",
        use_date_partition: bool = True,
        final_dir: Optional[str | Path] = None,  # 例: NAS mount
    ) -> None:
        self.base_dir = Path(base_dir)
        self.use_date_partition = bool(use_date_partition)
        self.final_dir = Path(final_dir) if final_dir is not None else None

    def make_run_context(
        self,
        *,
        prefix: str = "run",
        ensure_dirs: bool = False,
        now: Optional[datetime] = None,
    ) -> RunContext:
        """
        RunContext を生成する。

        ensure_dirs=False の場合:
          - ディレクトリ作成は一切しない（副作用ゼロ）
        ensure_dirs=True の場合:
          - tmp_dir だけ作る（out_dir 自体は save_* の atomic write で作られる）
        """
        now = now or datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S")
        run_id = f"{prefix}_{stamp}_{os.getpid()}"

        if self.use_date_partition:
            date_dir = now.strftime("%Y-%m-%d")
            out_dir = self.base_dir / date_dir / run_id
        else:
            out_dir = self.base_dir / run_id

        tmp_dir = out_dir / "_tmp"
        if ensure_dirs:
            tmp_dir.mkdir(parents=True, exist_ok=True)

        meta = {"run_id": run_id, "created_at_utc": _now_utc_iso()}
        return RunContext(run_id=run_id, out_dir=out_dir, tmp_dir=tmp_dir, meta=meta)

    def save_meta(
        self, ctx: RunContext, extra: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        meta.json を atomic に保存する。
        """
        meta = dict(ctx.meta)
        if extra:
            meta.update(extra)
        path = ctx.out_dir / "meta.json"
        _atomic_write_text(json.dumps(meta, ensure_ascii=False, indent=2), path)
        return path

    def save_jsonl(
        self, ctx: RunContext, *, rows: list[dict], name: str = "results.jsonl"
    ) -> Path:
        """
        JSONL を atomic に保存する。
        numpy / Path / datetime 等が混ざっても落ちない。
        """

        def _default(o: Any):
            # --- numpy scalar 対応 (np.float32, np.int64 等) ---
            try:
                import numpy as np  # type: ignore

                if isinstance(o, np.generic):
                    return o.item()
            except Exception:
                pass

            # --- Path 対応 ---
            from pathlib import Path

            if isinstance(o, Path):
                return str(o)

            # --- datetime など ---
            try:
                return o.isoformat()
            except Exception:
                pass

            # 最終 fallback
            return str(o)

        lines = (
            "\n".join(json.dumps(r, ensure_ascii=False, default=_default) for r in rows)
            + "\n"
        )

        path = ctx.out_dir / name
        _atomic_write_text(lines, path)
        return path

    def save_json(
        self,
        ctx: RunContext,
        *,
        obj: Dict[str, Any],
        name: str = "summary.json",
        indent: int = 2,
        sort_keys: bool = True,
    ) -> Path:
        """
        JSON を atomic に保存する（dict前提）。

        - make_run_context() ではディレクトリを作らない設計なので、
          実際の mkdir は _atomic_write_text() 内で行う。
        """
        text = json.dumps(
            obj,
            ensure_ascii=False,
            indent=indent,
            sort_keys=sort_keys,
        )
        path = ctx.out_dir / name
        _atomic_write_text(text + "\n", path)
        return path

    def save_text(
        self, ctx: RunContext, *, text: str, name: str, encoding: str = "utf-8"
    ) -> Path:
        """
        任意テキストを atomic に保存する（汎用）。
        """
        path = ctx.out_dir / name
        _atomic_write_text(text, path, encoding=encoding)
        return path

    def save_bytes(self, ctx: RunContext, *, data: bytes, name: str) -> Path:
        """
        任意バイナリを atomic に保存する（汎用）。
        """
        path = ctx.out_dir / name
        _atomic_write_bytes(data, path)
        return path

    def finalize_to_final_dir(self, ctx: RunContext) -> Optional[Path]:
        """
        final_dir が設定されている場合、ctx.out_dir を final 側へコピーする。
        """
        if self.final_dir is None:
            return None

        # base_dir からの相対を保って final にコピー（date partition を維持）
        try:
            rel = ctx.out_dir.relative_to(self.base_dir)
        except Exception:
            rel = Path(ctx.out_dir.name)

        dst = self.final_dir / rel
        _atomic_copytree(ctx.out_dir, dst)
        return dst
