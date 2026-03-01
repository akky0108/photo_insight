from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.core.batch_framework.utils.io_utils import group_by_key, write_csv_with_lock
from photo_insight.file_handler.exif_file_handler import ExifFileHandler

ExifData = Dict[str, str]


class NEFFileBatchProcess(BaseBatchProcessor):
    """RAW (NEF等) ファイルのバッチ処理クラス"""

    _DATE_RE_DASH = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    _DATE_RE_NODASH = re.compile(r"^\d{8}$")

    def __init__(self, config_path: Optional[str] = None, max_workers: int = 4):
        super().__init__(config_path=config_path, max_workers=max_workers)

        self.exif_handler = ExifFileHandler()

        self.exif_fields = self.config.get(
            "exif_fields",
            [
                "FileName",
                "Model",
                "Lens",
                "ISO",
                "Aperture",
                "FocalLength",
                "Rating",
                "ImageHeight",
                "ImageWidth",
                "Orientation",
                "BitDepth",
            ],
        )

        self.append_mode = bool(self.config.get("append_mode", False))
        self.output_directory = self.config.get("output_directory", "temp")

        base_dir = self.config.get("base_directory") or self.config.get("base_directory_root")
        if not base_dir:
            raise ValueError("config key 'base_directory' is required (or legacy 'base_directory_root').")
        self.base_directory_path = Path(base_dir)

        self._csv_locks: Dict[str, Lock] = defaultdict(Lock)

        self.output_data: List[Dict[str, Any]] = []
        self.success_count: int = 0
        self.failure_count: int = 0

        self.target_dirs: List[Path] = []
        self._nef_session_name: str = "ALL"

    # ------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------
    def setup(self) -> None:
        self.target_dirs = []

        self.output_data.clear()
        self.success_count = 0
        self.failure_count = 0

        session_name = "ALL"
        target_dir = getattr(self, "target_dir", None)
        if target_dir:
            session_name = Path(target_dir).name
        else:
            td = getattr(self, "target_date", None)
            if td:
                session_name = str(td).replace("/", "_")

        self._nef_session_name = session_name

        use_run_dir = bool(getattr(self, "_persist_run_results", False)) and getattr(self, "run_ctx", None) is not None
        if use_run_dir:
            self.temp_dir = Path(self.run_ctx.out_dir) / "artifacts" / "nef" / session_name
        else:
            self.temp_dir = Path(self.project_root) / self.output_directory / session_name

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"NEF output dir: {self.temp_dir}")

        base_dir = self.base_directory_path
        if not base_dir.exists():
            self.handle_error(f"ディレクトリが見つかりません: {base_dir}", raise_exception=True)

        target_dir = getattr(self, "target_dir", None)
        if target_dir:
            td = Path(target_dir)
            if not td.exists():
                self.handle_error(f"target_dir が見つかりません: {td}", raise_exception=True)
            self.target_dirs = [td]
            self.logger.info(f"初期設定完了: target_dir 指定のため単一セッションのみ (session={td.name})")
        else:
            target_date = getattr(self, "target_date", None)
            if target_date:
                self.target_dirs = self._find_date_dirs(base_dir, str(target_date))
                self.logger.info(f"初期設定完了: date={target_date} により sessions={len(self.target_dirs)}")
            else:
                self.target_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
                self.logger.info(f"初期設定完了: 画像ディレクトリ {base_dir} (sessions={len(self.target_dirs)})")

        super().setup()

    def cleanup(self) -> None:
        # ★ 0件でも latest は更新したい（incremental 全スキップに備える）
        if bool(getattr(self, "nef_incremental", False)):
            self._ensure_latest_for_session(self._nef_session_name)

        super().cleanup()
        self.logger.info("クリーンアップ完了")

    # ------------------------------------------------------------
    # data collection
    # ------------------------------------------------------------
    def load_data(self) -> List[Dict[str, Any]]:
        target_dir: Optional[Path] = getattr(self, "target_dir", None)
        session_name: Optional[str] = Path(target_dir).name if target_dir else None

        exts = getattr(self.exif_handler, "raw_extensions", [".NEF"])
        exts = [e.lower() for e in exts]

        def collect_files(dir_path: Path) -> List[Path]:
            files: List[Path] = []
            for ext in exts:
                files.extend(list(dir_path.rglob(f"*{ext}")))
                files.extend(list(dir_path.rglob(f"*{ext.upper()}")))
            uniq: Dict[str, Path] = {}
            for p in files:
                if p.is_file():
                    uniq[str(p.resolve())] = p
            return list(uniq.values())

        raw_files: List[Path] = []

        if target_dir:
            self.logger.info(f"指定ディレクトリのみ処理: {target_dir}")
            raw_files = collect_files(Path(target_dir))
            self.logger.info(f"{target_dir} から {len(raw_files)} 件検出")
        else:
            self.logger.info("全ディレクトリを対象に処理")
            for d in self.target_dirs:
                found = collect_files(d)
                self.logger.info(f"{d} から {len(found)} 件検出")
                raw_files.extend(found)

        nef_max_files = int(getattr(self, "nef_max_files", 0) or 0)
        if nef_max_files > 0:
            raw_files = raw_files[:nef_max_files]
            self.logger.info(f"nef_max_files applied: {nef_max_files}")

        if bool(getattr(self, "nef_dry_run", False)):
            self.logger.info(f"nef dry-run: detected {len(raw_files)} files (showing up to 20)")
            for p in raw_files[:20]:
                self.logger.info(f"dry-run target: {p}")
            return []

        data: List[Dict[str, Any]] = []
        for path in raw_files:
            subdir_name = session_name or path.parent.name
            data.append(
                {
                    "path": str(path),
                    "directory": str(path.parent),
                    "filename": path.name,
                    "subdir_name": subdir_name,
                }
            )

        self.logger.info(f"get_data(): 収集ファイル数 = {len(data)}")
        return data

    # ------------------------------------------------------------
    # batch processing
    # ------------------------------------------------------------
    def _generate_batches(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        grouped = group_by_key(data, "subdir_name")
        return list(grouped.values())

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not batch:
            return []

        grouped = group_by_key(batch, "subdir_name")
        all_exif_data: List[ExifData] = []

        incremental = bool(getattr(self, "nef_incremental", False))
        if incremental:
            done_dir = self._done_dir()
            done_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"nef incremental: done_dir={done_dir}")

        append_for_this_run = self.append_mode or incremental

        for subdir_name, items in grouped.items():
            exif_raw_list: List[Dict[str, Any]] = []
            processed_abs_paths: List[str] = []

            output_file_path = self.temp_dir / f"{subdir_name}_raw_exif_data.csv"

            if output_file_path.exists() and not self.append_mode and not incremental:
                self.logger.info(f"skip (already exists): {output_file_path}")
                continue

            for item in items:
                p = Path(str(item["path"]))
                abs_p = str(p.resolve())

                if incremental and self._done_marker(abs_p).exists():
                    self.logger.debug(f"skip (done): {p}")
                    continue

                exif_raw = self.exif_handler.read_file(str(p))
                if exif_raw:
                    exif_raw_list.append(exif_raw)
                    processed_abs_paths.append(abs_p)

            if not exif_raw_list:
                continue

            exif_data_list = self.filter_exif_data(exif_raw_list)

            write_csv_with_lock(
                file_path=output_file_path,
                data=exif_data_list,
                fieldnames=self.exif_fields,
                lock=self._get_lock_for_file(output_file_path),
                append=append_for_this_run,
                logger=self.logger,
            )

            # ★ stable pointer (runs/latest)
            self._update_latest_csv(subdir_name, output_file_path)

            if incremental:
                for abs_p in processed_abs_paths:
                    m = self._done_marker(abs_p)
                    if not m.exists():
                        m.write_text("done\n", encoding="utf-8")

            all_exif_data.extend(exif_data_list)

        with self.get_lock():
            self.output_data.extend(all_exif_data)
            self.success_count += len(all_exif_data)

        return [{"status": "success"} for _ in all_exif_data]

    # ------------------------------------------------------------
    # utils
    # ------------------------------------------------------------
    def filter_exif_data(self, raw_files: List[Dict[str, Any]]) -> List[ExifData]:
        results: List[ExifData] = []
        for raw in raw_files:
            filtered: ExifData = {field: str(raw.get(field, "N/A")) for field in self.exif_fields}
            results.append(filtered)
        return results

    def _get_lock_for_file(self, file_path: Path) -> Lock:
        key = str(file_path.resolve())
        return self._csv_locks[key]

    # ------------------------------------------------------------
    # date filter helpers (minimal)
    # ------------------------------------------------------------
    def _norm_date(self, s: str) -> str:
        s = s.strip()
        if self._DATE_RE_DASH.match(s):
            return s
        if self._DATE_RE_NODASH.match(s):
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        raise ValueError(f"Invalid date: {s} (YYYY-MM-DD or YYYYMMDD)")

    def _find_date_dirs(self, base_dir: Path, date_str: str) -> List[Path]:
        norm = self._norm_date(date_str)
        y = norm[:4]
        year_dir = base_dir / y

        candidates = [
            year_dir / norm,
            year_dir / norm.replace("-", ""),
        ]

        if base_dir.name == y:
            candidates += [base_dir / norm, base_dir / norm.replace("-", "")]

        out: List[Path] = []
        seen: set[str] = set()
        for p in candidates:
            if p.is_dir():
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    out.append(p)
        return out

    # ------------------------------------------------------------
    # incremental done marker (minimal & stable across run_id)
    # ------------------------------------------------------------
    def _done_dir(self) -> Path:
        out_root = Path(self.config.get("paths", {}).get("output_data_dir", "/work/output"))
        return out_root / "_done_nef"

    def _done_marker(self, abs_path: str) -> Path:
        h = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()
        return self._done_dir() / f"{h}.done"

    # ------------------------------------------------------------
    # runs root (IMPORTANT): /work/runs を正にする
    # ------------------------------------------------------------
    def _runs_root(self) -> Path:
        # 優先: config(debug.run_results_dir) -> env(PHOTO_INSIGHT_OUTPUT_DIR) -> /work/runs
        debug_dir = (self.config.get("debug", {}) or {}).get("run_results_dir")
        if debug_dir:
            return Path(str(debug_dir))
        env_dir = os.getenv("PHOTO_INSIGHT_OUTPUT_DIR")
        if env_dir:
            return Path(env_dir)
        return Path("/work/runs")

    # ------------------------------------------------------------
    # latest pointer (stable path) - copy (phase1)
    # ------------------------------------------------------------
    def _latest_dir(self) -> Path:
        return self._runs_root() / "latest" / "nef"

    def _update_latest_csv(self, session_name: str, src_csv: Path) -> None:
        dst_dir = self._latest_dir() / session_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_csv = dst_dir / src_csv.name
        try:
            shutil.copy2(src_csv, dst_csv)
            self.logger.info(f"latest updated: {dst_csv}")
        except Exception as e:
            self.logger.warning(f"failed to update latest csv: src={src_csv} dst={dst_csv} err={e}")

    def _ensure_latest_for_session(self, session_name: str) -> None:
        # 1) 今回の temp_dir にCSVがあるならそれを優先
        candidate = self.temp_dir / f"{session_name}_raw_exif_data.csv"
        if candidate.exists():
            self._update_latest_csv(session_name, candidate)
            return

        # 2) 過去runの artifacts から session のCSVを探して、mtime最新を採用
        runs_root = self._runs_root()
        pattern = f"**/artifacts/nef/{session_name}/{session_name}_raw_exif_data.csv"
        found = list(runs_root.glob(pattern))

        if not found:
            self.logger.info(f"latest unchanged: no existing csv found for session={session_name}")
            return

        latest_src = max(found, key=lambda p: p.stat().st_mtime)
        self._update_latest_csv(session_name, latest_src)
