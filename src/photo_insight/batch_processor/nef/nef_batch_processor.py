# src/photo_insight/batch_processor/nef_batch_process.py
from __future__ import annotations

from typing import List, Dict, Optional, Any
from pathlib import Path
from threading import Lock
from collections import defaultdict

from photo_insight.file_handler.exif_file_handler import ExifFileHandler
from photo_insight.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.batch_framework.utils.io_utils import (
    group_by_key,
    write_csv_with_lock,
)

ExifData = Dict[str, str]


class NEFFileBatchProcess(BaseBatchProcessor):
    """RAW (NEF等) ファイルのバッチ処理クラス"""

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

        # config: base_directory を正とし、互換として base_directory_root も許可
        base_dir = self.config.get("base_directory") or self.config.get(
            "base_directory_root"
        )
        if not base_dir:
            # 安定運用優先：意図しない場所を探索しない
            raise ValueError(
                "config key 'base_directory' is required (or legacy 'base_directory_root')."
            )
        self.base_directory_path = Path(base_dir)

        self._csv_locks: Dict[str, Lock] = defaultdict(Lock)

        # 集計用（必要なら後段で使う）
        self.output_data: List[Dict[str, Any]] = []
        self.success_count: int = 0
        self.failure_count: int = 0

        # Base契約: setup() で self.data が作られる（load_data() が呼ばれる）
        self.target_dirs: List[Path] = []

    # ------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------
    def setup(self) -> None:
        # ★ 必須：常に定義（Base.setup() から load_data() が呼ばれるため）
        self.target_dirs = []

        self.output_data.clear()
        self.success_count = 0
        self.failure_count = 0

        # ----------------------------
        # 出力先ディレクトリ決定
        # runs/ 配下に閉じる（FW安定運用優先）
        # ----------------------------
        session_name = "ALL"
        target_dir = getattr(self, "target_dir", None)
        if target_dir:
            session_name = Path(target_dir).name  # 例: 2026-02-17

        # run_ctx があり、かつ persist_run_results が有効なら runs 配下へ
        use_run_dir = (
            bool(getattr(self, "_persist_run_results", False))
            and getattr(self, "run_ctx", None) is not None
        )
        if use_run_dir:
            # 例: runs/YYYY-MM-DD/<run_id>/artifacts/nef/<session>
            self.temp_dir = (
                Path(self.run_ctx.out_dir) / "artifacts" / "nef" / session_name
            )
        else:
            # フォールバック（従来通り）
            self.temp_dir = (
                Path(self.project_root) / self.output_directory / session_name
            )

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"NEF output dir: {self.temp_dir}")

        base_dir = self.base_directory_path
        if not base_dir.exists():
            self.handle_error(
                f"ディレクトリが見つかりません: {base_dir}", raise_exception=True
            )

        target_dir = getattr(self, "target_dir", None)
        if target_dir:
            # target_dir 指定時はそれだけに絞る（安定運用・ログ明確化）
            td = Path(target_dir)
            if not td.exists():
                self.handle_error(
                    f"target_dir が見つかりません: {td}", raise_exception=True
                )
            self.target_dirs = [td]
            self.logger.info(
                f"初期設定完了: target_dir 指定のため単一セッションのみ (session={td.name})"
            )
        else:
            # 直下のディレクトリを対象とする（撮影セッション単位の想定）
            self.target_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
            self.logger.info(
                f"初期設定完了: 画像ディレクトリ {base_dir} (sessions={len(self.target_dirs)})"
            )

        # Base契約: setup() -> self.data = self.get_data() -> after_data_loaded(self.data)
        super().setup()

    def cleanup(self) -> None:
        super().cleanup()
        self.logger.info("クリーンアップ完了")

    # ------------------------------------------------------------
    # data collection
    # ------------------------------------------------------------
    def load_data(self) -> List[Dict[str, Any]]:
        """
        BaseBatchProcessor 契約:
        - load_data(): 純I/O（副作用なし）
        - キャッシュは Base が握る（get_data() は Base 側）

        仕様:
        - CLI(run_batch) 等で self.target_dir が設定されていれば、そのディレクトリのみ処理
        - 未設定なら setup() で収集した self.target_dirs を全対象に処理
        """
        target_dir: Optional[Path] = getattr(self, "target_dir", None)
        session_name: Optional[str] = Path(target_dir).name if target_dir else None

        # ExifFileHandler の拡張子定義を優先（.NEF 固定をやめる）
        exts = getattr(self.exif_handler, "raw_extensions", [".NEF"])
        exts = [e.lower() for e in exts]

        def collect_files(dir_path: Path) -> List[Path]:
            files: List[Path] = []
            # 拡張子ごとに探索（rglob("*")より少し安全・意図が明確）
            for ext in exts:
                files.extend(list(dir_path.rglob(f"*{ext}")))
                # 念のため大文字も拾う（ext=".nef" の場合でも "* .NEF" を拾える）
                files.extend(list(dir_path.rglob(f"*{ext.upper()}")))
            # 重複排除（同一パスが2回入る可能性がある）
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

        data: List[Dict[str, Any]] = []
        for path in raw_files:
            # Base/テスト/ログ/シリアライズで扱いやすいよう、path は str を正とする
            # ★重要: target_dir 指定時は subdir_name を session_name に固定（ネストしても撮影日単位に揃う）
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
    def _generate_batches(
        self, data: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        grouped = group_by_key(data, "subdir_name")
        return list(grouped.values())

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not batch:
            return []

        grouped = group_by_key(batch, "subdir_name")
        all_exif_data: List[ExifData] = []

        for subdir_name, items in grouped.items():
            exif_raw_list: List[Dict[str, Any]] = []

            output_file_path = self.temp_dir / f"{subdir_name}_raw_exif_data.csv"

            # ★ 永続キャッシュ：既に出力済みならスキップ（append_mode=False のとき）
            if output_file_path.exists() and not self.append_mode:
                self.logger.info(f"skip (already exists): {output_file_path}")
                continue

            for item in items:
                exif_raw = self.exif_handler.read_file(str(item["path"]))
                if exif_raw:
                    exif_raw_list.append(exif_raw)

            if not exif_raw_list:
                continue

            exif_data_list = self.filter_exif_data(exif_raw_list)

            write_csv_with_lock(
                file_path=output_file_path,
                data=exif_data_list,
                fieldnames=self.exif_fields,
                lock=self._get_lock_for_file(output_file_path),
                append=self.append_mode,
                logger=self.logger,
            )

            all_exif_data.extend(exif_data_list)

        with self.get_lock():
            self.output_data.extend(all_exif_data)
            self.success_count += len(all_exif_data)

        # Base側の集計に乗せるため、1件ごとに status を返す
        return [{"status": "success"} for _ in all_exif_data]

    # ------------------------------------------------------------
    # utils
    # ------------------------------------------------------------
    def filter_exif_data(self, raw_files: List[Dict[str, Any]]) -> List[ExifData]:
        results: List[ExifData] = []
        for raw in raw_files:
            filtered: ExifData = {
                field: str(raw.get(field, "N/A")) for field in self.exif_fields
            }
            results.append(filtered)
        return results

    def _get_lock_for_file(self, file_path: Path) -> Lock:
        key = str(file_path.resolve())
        return self._csv_locks[key]
