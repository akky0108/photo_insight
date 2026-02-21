# src/photo_insight/batch_processor/portrait_quality/portrait_quality_batch_processor.py
from __future__ import annotations

import os
import csv
import gc
from pathlib import Path
from threading import Event
from typing import Any, List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError

from photo_insight.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.image_loader import ImageLoader
from photo_insight.evaluators.portrait_quality.portrait_quality_evaluator import (
    PortraitQualityEvaluator,
)
from photo_insight.portrait_quality_header import PortraitQualityHeaderGenerator
from photo_insight.monitoring.memory_monitor import MemoryMonitor


class PortraitQualityBatchProcessor(BaseBatchProcessor):
    """ポートレート写真の品質評価を行うバッチ処理クラス"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        logger: Optional[object] = None,
        max_workers: Optional[int] = None,
        date: Optional[str] = None,
        batch_size: Optional[int] = None,
        # ===== Config DI (Baseへ委譲) =====
        config_env: Optional[str] = None,
        config_paths: Optional[List[str]] = None,
        resolver: Any = None,
        loader: Any = None,
        watch_factory: Any = None,
        list_policy: str = "replace",
        strict_missing: bool = True,
        auto_load: bool = True,
    ):
        super().__init__(
            config_path=config_path,
            config_env=config_env,
            config_paths=config_paths,
            max_workers=max_workers or 1,
            logger=None,  # Base側で生成（後で差し替え可能）
            resolver=resolver,
            loader=loader,
            watch_factory=watch_factory,
            list_policy=list_policy,
            strict_missing=strict_missing,
            auto_load=auto_load,
        )

        # config / logger
        self.config = self.config_manager.get_config()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = self.config_manager.get_logger("PortraitQualityBatchProcessor")

        self.memory_monitor = MemoryMonitor(self.logger)
        self.date = date
        self.processed_images: set[str] = set()
        self.image_loader = ImageLoader(logger=self.logger)

        # processed_images のファイル追記 & set 更新を守るロック（基底と統一）
        self._processed_lock = self.get_lock()

        self.image_csv_file: Optional[str] = None
        self.result_csv_file: Optional[str] = None
        self.processed_images_file: Optional[str] = None
        self.output_directory: Optional[str] = None
        self.base_directory: Optional[str] = None

        # 互換フラグ
        self.completed_all_batches = False
        self.memory_threshold_exceeded = False

        self.batch_size = batch_size or self.config.get("batch_size", 10)
        self.memory_threshold = self.config_manager.get_memory_threshold(default=90)

        # stop control
        self._stop_event = Event()

        # run bookkeeping
        self._start_processed_count: int = 0
        self._total_images_to_process: int = 0

        # --- Baseが拾う optional ---
        # self.stop_reason: Optional[str] = None
        # self.processed_count_this_run: Optional[int] = None

    def on_config_change(self, new_config: dict) -> None:
        self.logger.info("Config updated. Applying new settings...")
        self.batch_size = new_config.get("batch_size", self.batch_size)

        out_dir = new_config.get("output_directory") or self.output_directory or "temp"
        self.output_directory = out_dir
        os.makedirs(self.output_directory, exist_ok=True)

    # ============================================================
    # stop control / memory
    # ============================================================
    def _check_and_maybe_stop(self, where: str) -> bool:
        """
        メモリ使用量が閾値を超えたら停止フラグを立てる。
        True: stopした（or stopすべき）
        """
        try:
            mem_usage = float(self.memory_monitor.get_memory_usage())
        except Exception:
            return False

        if mem_usage > float(self.memory_threshold):
            if not getattr(self, "memory_threshold_exceeded", False):
                self.logger.warning(
                    f"Memory usage {mem_usage:.1f}% exceeded threshold "
                    f"({self.memory_threshold}%) at {where}. Stopping."
                )
            self.memory_threshold_exceeded = True

            # ★ Base(summary.json) にも stop理由を載せたいならここで決める
            # （語彙はPQ側が所有。Baseは拾って記録するだけ）
            if not getattr(self, "stop_reason", None):
                self.stop_reason = "memory_threshold"

            self._stop_event.set()
            return True
        return False

    def _stopping(self) -> bool:
        """stop状態（静かに即returnするための統一判定）"""
        return bool(
            self._stop_event.is_set()
            or getattr(self, "memory_threshold_exceeded", False)
        )

    # ============================================================
    # paths
    # ============================================================
    def _resolve_session_name(self) -> str:
        td = getattr(self, "target_dir", None)
        if td:
            return Path(td).name
        if getattr(self, "date", None):
            return str(self.date)
        return "ALL"

    def _resolve_nef_input_csv(self) -> Optional[str]:
        session = self._resolve_session_name()
        fname = f"{session}_raw_exif_data.csv"

        # 1) 同一run内（あれば最優先）
        if getattr(self, "run_ctx", None) is not None:
            p1 = Path(self.run_ctx.out_dir) / "artifacts" / "nef" / session / fname
            if p1.exists():
                return str(p1)

        # 2) runs/latest（運用の正）
        p2 = Path(self.project_root) / "runs" / "latest" / "nef" / session / fname
        if p2.exists():
            return str(p2)

        return None

    def _resolve_output_directory(self) -> Path:
        session = self._resolve_session_name()
        use_run_dir = (
            bool(getattr(self, "_persist_run_results", False))
            and getattr(self, "run_ctx", None) is not None
        )
        if use_run_dir:
            out_dir = (
                Path(self.run_ctx.out_dir)
                / "artifacts"
                / "portrait_quality"
                / session
            )
        else:
            base = self.config.get("output_directory", "temp")
            out_dir = Path(self.project_root) / base / session

        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ============================================================
    # lifecycle
    # ============================================================
    def setup(self) -> None:
        self.logger.info("Setting up PortraitQualityBatchProcessor...")

        # ★ Stop flags (per-run)
        self._stop_event = Event()
        self.memory_threshold_exceeded = False
        self.completed_all_batches = False

        # ★ per-run optional fields for Base summary
        self.stop_reason = None
        self.processed_count_this_run = None

        # 1) 出力先を最初に確定（runs/artifacts）
        out_dir = self._resolve_output_directory()
        self.output_directory = str(out_dir)
        self.logger.info(f"PortraitQuality output dir: {self.output_directory}")

        # 2) ファイルパス・入力元の確定
        self._set_directories_and_files()

        # 3) 既処理ロード
        self._load_processed_images()
        self._start_processed_count = len(self.processed_images)

        # Base契約:
        super().setup()

    def _set_directories_and_files(self) -> None:
        import re

        picture_root = self.config.get("picture_root", "/mnt/l/picture")
        out_dir = self.output_directory or self.config.get("output_directory", "temp")
        os.makedirs(out_dir, exist_ok=True)

        # --- base_directory (target_dir > date > config) ---
        target_dir = getattr(self, "target_dir", None)
        if target_dir:
            self.base_directory = str(Path(target_dir))
        elif self.date:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date):
                raise ValueError(f"Invalid date format: {self.date} (expected YYYY-MM-DD)")
            year = self.date[:4]
            self.base_directory = os.path.join(picture_root, year, self.date)
        else:
            self.base_directory = self.config.get(
                "base_directory", "/mnt/l/picture/2025/2025-01-01"
            )

        if not self.base_directory or not os.path.isdir(self.base_directory):
            raise FileNotFoundError(f"Base directory does not exist: {self.base_directory}")

        # --- input exif csv (B) ---
        nef_csv = self._resolve_nef_input_csv()
        if nef_csv is None:
            raise FileNotFoundError(
                f"NEF exif CSV not found for session={self._resolve_session_name()}. "
                f"Expected under runs/latest/nef/<session>/ or same run artifacts."
            )
        self.image_csv_file = nef_csv
        self.logger.info(f"Input NEF CSV: {self.image_csv_file}")

        # --- output files (session-based) ---
        session = self._resolve_session_name()
        self.result_csv_file = os.path.join(out_dir, f"evaluation_results_{session}.csv")
        self.processed_images_file = os.path.join(out_dir, f"processed_images_{session}.txt")

        self.logger.info(f"ベースディレクトリ: {self.base_directory}")

    def _load_processed_images(self) -> None:
        if self.processed_images_file and os.path.exists(self.processed_images_file):
            with open(self.processed_images_file, "r", encoding="utf-8") as f:
                self.processed_images = set(f.read().splitlines())
            self.logger.info(f"Loaded {len(self.processed_images)} previously processed images.")

        if self.result_csv_file and os.path.exists(self.result_csv_file):
            self.logger.info(
                f"Result file exists: {self.result_csv_file} — continuing from previous run."
            )

    # ============================================================
    # data loading
    # ============================================================
    def load_image_data(self) -> List[Dict[str, str]]:
        # --- (B) resolve input NEF CSV automatically ---
        if (not self.image_csv_file) or (not os.path.exists(self.image_csv_file)):
            nef_csv = self._resolve_nef_input_csv()
            if nef_csv is not None:
                self.image_csv_file = nef_csv
                self.logger.info(f"[B] Using NEF exif CSV: {self.image_csv_file}")
            else:
                self.logger.error(
                    f"[B] NEF exif CSV not found for session={self._resolve_session_name()} "
                    f"(checked same-run and runs/latest)."
                )
                return []

        self.logger.info(f"Loading image data from {self.image_csv_file}")
        image_data: List[Dict[str, str]] = []
        try:
            with open(self.image_csv_file, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                required_keys = {"FileName", "Orientation", "BitDepth"}

                if not required_keys.issubset(set(reader.fieldnames or [])):
                    self.logger.error(f"Missing required columns in CSV. Found: {reader.fieldnames}")
                    return []

                for row in reader:
                    image_data.append(
                        {
                            "file_name": row["FileName"],
                            "orientation": row["Orientation"],
                            "bit_depth": row["BitDepth"],
                        }
                    )
        except FileNotFoundError:
            self.logger.warning(f"File not found: {self.image_csv_file}. Proceeding with empty data.")
        except csv.Error as e:
            self.logger.error(f"CSV parsing error in {self.image_csv_file}: {e}")
        return image_data

    def after_data_loaded(self, data: List[Dict]) -> None:
        self._total_images_to_process = len(data)
        if self.processed_images:
            self.logger.info(
                f"Found {len(self.processed_images)} previously processed images. Resuming from there."
            )
        else:
            self.logger.info("No previously processed images found. Starting fresh.")

        self.logger.info(f"Memory usage threshold set to {self.memory_threshold}% from config.")
        self.logger.info(f"Total images to process: {self._total_images_to_process}")

    def load_data(self) -> List[Dict[str, str]]:
        raw_data = self.load_image_data()
        with self._processed_lock:
            return [d for d in raw_data if d.get("file_name") not in self.processed_images]

    # ============================================================
    # batch processing
    # ============================================================
    def _process_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        # ★ stop中は静かに即return（Baseが次バッチを呼んでもログスパムしない）
        if self._stopping():
            return []

        # バッチ開始時点で止める（次バッチを出さない）
        if self._check_and_maybe_stop("batch_start"):
            return []

        # 未処理画像のみ残す
        with self._processed_lock:
            batch = [img for img in batch if img.get("file_name") not in self.processed_images]

        if not batch:
            self.logger.debug("All images in this batch are already processed. Skipping.")
            return []

        # 並列／直列処理
        if (self.max_workers or 1) == 1:
            results = self._process_batch_serial(batch)
        else:
            results = self._process_batch_parallel(batch)

        # CSVはフル結果で保存
        if results and self.result_csv_file:
            self.save_results(results, self.result_csv_file)

        # GC + メモリログ（重い処理の後に一回だけ）
        gc.collect()
        self.memory_monitor.log_usage(prefix="Post batch GC")
        self._check_and_maybe_stop("batch_end")

        # ★Baseに返すのはJSON安全なmini（results.jsonl事故を防ぐ）
        mini: List[Dict[str, Any]] = []
        for r in results or []:
            score = r.get("overall_score", None)
            try:
                score_f = float(score) if score not in ("", None) else None
            except Exception:
                score_f = None

            mini.append(
                {
                    "status": "success",
                    "file_name": r.get("file_name"),
                    "score": score_f,
                }
            )

        return mini

    def process_image(
        self, file_name: str, orientation: str, bit_depth: str
    ) -> Optional[Dict[str, Union[str, float, bool]]]:
        if self._stop_event.is_set():
            return None
        if self._check_and_maybe_stop("worker_start"):
            return None

        self.logger.info(f"Processing image: {file_name}")
        result: Dict[str, Any] = {
            "file_name": os.path.basename(file_name),
            "sharpness_score": None,
            "blurriness_score": None,
            "contrast_score": None,
            "noise_score": None,
            "local_sharpness_score": None,
            "local_sharpness_std": None,
            "local_contrast_score": None,
            "local_contrast_std": None,
            "full_body_detected": None,
            "pose_score": None,
            "headroom_ratio": None,
            "footroom_ratio": None,
            "side_margin_min_ratio": None,
            "full_body_cut_risk": None,
            "accepted_flag": None,
            "accepted_reason": None,
            "delta_face_sharpness": None,
            "delta_face_contrast": None,
            "lead_room_score": None,
            "face_detected": None,
            "faces": None,
            "face_sharpness_score": None,
            "face_contrast_score": None,
            "face_noise_score": None,
            "face_local_sharpness_score": None,
            "face_local_sharpness_std": None,
            "face_local_contrast_score": None,
            "face_local_contrast_std": None,
        }

        try:
            image = self.image_loader.load_image(file_name, orientation, bit_depth)
            evaluator = PortraitQualityEvaluator(
                image_input=image,
                is_raw=False,
                logger=self.logger,
                file_name=file_name,
                config_manager=self.config_manager,
                quality_profile=self.config.get("quality_profile", "portrait"),
                thresholds_path=self.config.get("evaluator_thresholds_path"),
            )
            eval_result = evaluator.evaluate()

            del image
            del evaluator

            if eval_result:
                result.update(eval_result)
                return result
            self.logger.warning(f"No evaluation result for image {file_name}")
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_name}")
        except Exception as e:
            self.logger.error(f"Error processing image {file_name}: {e}", exc_info=True)
        return None

    def _process_single_image(self, img_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
        if self._stop_event.is_set():
            return None

        try:
            if not self.base_directory:
                raise RuntimeError("base_directory is not set")

            result = self.process_image(
                os.path.join(self.base_directory, img_info["file_name"]),
                img_info["orientation"],
                img_info["bit_depth"],
            )
            if result:
                self.logger.info(f"Processed image: {img_info['file_name']}")
                self._mark_as_processed(img_info["file_name"])
                return result
        except Exception as e:
            self.logger.error(
                f"Failed to process image {img_info.get('file_name')}: {e}",
                exc_info=True,
            )
        return None

    def _process_batch_serial(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for img_info in batch:
            if self._stop_event.is_set():
                break
            if self._check_and_maybe_stop("serial_loop"):
                break

            result = self._process_single_image(img_info)
            if result:
                results.append(result)
        return results

    def _process_batch_parallel(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers or 2) as executor:
            futures = []

            for img_info in batch:
                if self._stop_event.is_set():
                    break
                if self._check_and_maybe_stop("parallel_before_submit"):
                    break
                futures.append(executor.submit(self._process_single_image, img_info))

            cancelled_pending = False
            for future in as_completed(futures):
                if self._stop_event.is_set() and not cancelled_pending:
                    cancelled_pending = True
                    for f in futures:
                        f.cancel()

                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except CancelledError:
                    continue
                except Exception:
                    self.logger.exception("[Parallel] Failed to process image", exc_info=True)
                    continue

        return results

    # ============================================================
    # persistence
    # ============================================================
    def _mark_as_processed(self, file_name: str) -> None:
        if not self.processed_images_file:
            raise RuntimeError("processed_images_file is not set")

        with self._processed_lock:
            if file_name in self.processed_images:
                return

            with open(self.processed_images_file, "a", encoding="utf-8") as f:
                f.write(f"{file_name}\n")
                f.flush()
                os.fsync(f.fileno())

            self.processed_images.add(file_name)
            self.logger.debug(f"Marked as processed: {file_name}")

    def save_results(self, results: List[Dict[str, Union[str, float, bool]]], file_path: str) -> None:
        self.logger.info(f"Saving results to {file_path}")
        file_exists = os.path.isfile(file_path)
        results_sorted = sorted(results, key=lambda x: str(x.get("file_name", "")))

        header_generator = PortraitQualityHeaderGenerator()
        fieldnames = header_generator.get_all_headers()

        with self.get_lock():
            with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                for row in results_sorted:
                    writer.writerow({key: row.get(key, None) for key in fieldnames})

    # ============================================================
    # cleanup
    # ============================================================
    def cleanup(self) -> None:
        super().cleanup()
        self.logger.info("Cleaning up PortraitQualityBatchProcessor-specific resources.")

        processed_this_run = len(self.processed_images) - int(self._start_processed_count or 0)
        remaining_count = int(self._total_images_to_process or 0) - processed_this_run
        if remaining_count < 0:
            remaining_count = 0

        # ★ Baseが拾って summary.json に入れる（ドメイン上の“実処理数”はPQが定義）
        self.processed_count_this_run = int(processed_this_run)

        if getattr(self, "memory_threshold_exceeded", False):
            # stop_reason が未設定なら、ここでも保険でセット
            if not getattr(self, "stop_reason", None):
                self.stop_reason = "memory_threshold"

            self.logger.warning(
                f"Batch processing stopped due to memory threshold. "
                f"Processed {processed_this_run} images, {remaining_count} remaining. "
                "You can re-run to continue remaining images."
            )
        elif getattr(self, "completed_all_batches", False):
            self.logger.info(
                f"All batches processed successfully. Total processed images: {processed_this_run}"
            )

        # ★重要: stopした場合は processed_images_file を消さない（再開に必要）
        processed_file = getattr(self, "processed_images_file", None)
        if (
            getattr(self, "completed_all_batches", False)
            and not getattr(self, "memory_threshold_exceeded", False)
            and processed_file
            and os.path.exists(processed_file)
        ):
            self.logger.info(f"Removing processed images file: {processed_file}")
            os.remove(processed_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process images with PortraitQualityBatchProcessor."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "Config file path (optional). If omitted, ConfigManager uses "
            "CONFIG_ENV / defaults."
        ),
    )
    parser.add_argument(
        "--config_env",
        type=str,
        default=None,
        help=(
            "Config environment (e.g. prod/test). If omitted, "
            "CONFIG_ENV env-var may be used."
        ),
    )
    parser.add_argument(
        "--config_paths",
        nargs="*",
        default=None,
        help=("Optional explicit config file list (applied in order; supports extends)."),
    )
    parser.add_argument("--date", type=str, help="Specify the date for directory and file names.")
    parser.add_argument("--max_workers", type=int, help="Number of worker threads")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    args = parser.parse_args()
    print("[DEBUG] args:", args)

    processor = PortraitQualityBatchProcessor(
        config_path=args.config_path,
        config_env=args.config_env,
        config_paths=args.config_paths,
        date=args.date,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )
    processor.execute()
