from __future__ import annotations

import csv
import gc
import os
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, Union

from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.evaluators.portrait_quality.portrait_quality_evaluator import (
    PortraitQualityEvaluator,
)
from photo_insight.image_loader import ImageLoader
from photo_insight.monitoring.memory_monitor import MemoryMonitor
from photo_insight.portrait_quality_header import PortraitQualityHeaderGenerator


class PortraitQualityBatchProcessor(BaseBatchProcessor):
    """ポートレート写真の品質評価を行うバッチ処理クラス"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        logger: Optional[object] = None,
        max_workers: Optional[int] = None,
        date: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_images: Optional[int] = None,
        input_csv_path: Optional[str] = None,
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
            logger=None,
            resolver=resolver,
            loader=loader,
            watch_factory=watch_factory,
            list_policy=list_policy,
            strict_missing=strict_missing,
            auto_load=auto_load,
        )

        self.config = self.config_manager.get_config()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = self.config_manager.get_logger("PortraitQualityBatchProcessor")

        self.memory_monitor = MemoryMonitor(self.logger)
        self.date = date
        self.max_images = max_images
        self.input_csv_path = input_csv_path

        self.processed_images: set[str] = set()
        self.image_loader = ImageLoader(logger=self.logger)

        self._processed_lock = self.get_lock()

        self.image_csv_file: Optional[str] = None
        self.result_csv_file: Optional[str] = None
        self.processed_images_file: Optional[str] = None
        self.output_directory: Optional[str] = None
        self.base_directory: Optional[str] = None

        self.completed_all_batches = False
        self.memory_threshold_exceeded = False

        self.batch_size = batch_size or self.config.get("batch_size", 10)
        self.memory_threshold = self.config_manager.get_memory_threshold(default=90)

        self._stop_event = Event()

        self._start_processed_count: int = 0
        self._total_images_to_process: int = 0

        self.processed_count_this_run: Optional[int] = None

    def _dbg(self, message: str) -> None:
        print(f"[PQ-DBG] {message}", flush=True)
        try:
            self.logger.info(f"[PQ-DBG] {message}")
        except Exception:
            pass

    # ============================================================
    # run 条件の事前反映
    # ============================================================
    def execute(
        self,
        *args: Any,
        max_images: Optional[int] = None,
        input_csv_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        setup() より前に run 固有条件を反映する。

        Parameters
        ----------
        max_images : Optional[int]
            この run における処理上限。

        input_csv_path : Optional[str]
            前段 NEF stage が生成した入力 CSV パス。
            指定された場合はこの CSV を最優先で使用する。
        """
        if max_images is not None:
            self.max_images = max_images

        if input_csv_path is not None:
            self.input_csv_path = str(input_csv_path)

        original_max_workers = self.max_workers
        try:
            if self.max_images is not None:
                self.max_workers = 1
                self.logger.info(
                    "max_images mode enabled. forcing single-worker execution for strict stop control "
                    f"(max_images={self.max_images})."
                )

            return super().execute(*args, **kwargs)
        finally:
            self.max_workers = original_max_workers

    def process(
        self,
        *args: Any,
        max_images: Optional[int] = None,
        input_csv_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        互換用:
        BaseBatchProcessor.process() が受け取らない kwargs を、
        Processor 側で吸収して破綻しないようにする。
        """
        if max_images is not None:
            self.max_images = max_images

        if input_csv_path is not None:
            self.input_csv_path = str(input_csv_path)

        return super().process(*args, **kwargs)

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

            if not getattr(self, "stop_reason", None):
                self.stop_reason = "memory_threshold"

            self._stop_event.set()
            return True
        return False

    def _stopping(self) -> bool:
        """stop状態（静かに即returnするための統一判定）"""
        return bool(self._stop_event.is_set() or getattr(self, "memory_threshold_exceeded", False))

    def _should_stop_by_max_images(self) -> bool:
        """
        max_images による途中停止判定。
        主目的は「途中停止 → 次回 resume」。
        """
        if self.max_images is None:
            return False

        try:
            limit = int(self.max_images)
        except Exception as e:
            raise ValueError(f"Invalid max_images: {self.max_images}") from e

        if limit < 0:
            raise ValueError(f"max_images must be >= 0. got: {limit}")

        if limit == 0:
            if not getattr(self, "stop_reason", None):
                self.stop_reason = "max_images_limit"
            self._stop_event.set()
            return True

        current = int(self.processed_count_this_run or 0)
        if current >= limit:
            if not getattr(self, "stop_reason", None):
                self.stop_reason = "max_images_limit"
            self._stop_event.set()
            return True

        return False

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
        if self.input_csv_path:
            explicit = Path(self.input_csv_path)
            if not explicit.exists():
                raise FileNotFoundError(f"Explicit input_csv_path not found: {explicit}")

            self.logger.info(f"Using explicit input CSV: {explicit}")
            return str(explicit)

        session = self._resolve_session_name()
        fname = f"{session}_raw_exif_data.csv"

        self.logger.info(f"Resolving NEF CSV: session={session}, project_root={self.project_root}")

        if getattr(self, "run_ctx", None) is not None:
            p1 = Path(self.run_ctx.out_dir) / "artifacts" / "nef" / session / fname
            if p1.exists():
                return str(p1)

        p2 = Path(self.project_root) / "runs" / "latest" / "nef" / session / fname
        if p2.exists():
            return str(p2)

        return None

    def _resolve_output_directory(self) -> Path:
        session = self._resolve_session_name()

        if self.max_images is not None:
            out_dir = Path(self.project_root) / "runs" / "latest" / "portrait_quality" / session
        else:
            use_run_dir = (
                bool(getattr(self, "_persist_run_results", False)) and getattr(self, "run_ctx", None) is not None
            )
            if use_run_dir:
                out_dir = Path(self.run_ctx.out_dir) / "artifacts" / "portrait_quality" / session
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

        self._stop_event = Event()
        self.memory_threshold_exceeded = False
        self.completed_all_batches = False

        self.stop_reason = None
        self.processed_count_this_run = 0

        self._should_stop_by_max_images()

        out_dir = self._resolve_output_directory()
        self.output_directory = str(out_dir)
        self.logger.info(f"PortraitQuality output dir: {self.output_directory}")

        self._set_directories_and_files()

        self._load_processed_images()
        self._start_processed_count = len(self.processed_images)

        super().setup()

    def _set_directories_and_files(self) -> None:
        import re

        picture_root = self.config.get("picture_root", "/mnt/l/picture")
        out_dir = self.output_directory or self.config.get("output_directory", "temp")
        os.makedirs(out_dir, exist_ok=True)

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
                "base_directory",
                "/mnt/l/picture/2025/2025-01-01",
            )

        if not self.base_directory or not os.path.isdir(self.base_directory):
            raise FileNotFoundError(f"Base directory does not exist: {self.base_directory}")

        nef_csv = self._resolve_nef_input_csv()
        if nef_csv is None:
            raise FileNotFoundError(
                f"NEF exif CSV not found for session={self._resolve_session_name()}. "
                f"Expected under runs/latest/nef/<session>/ or same run artifacts."
            )
        self.image_csv_file = nef_csv
        self.logger.info(f"Input NEF CSV: {self.image_csv_file}")

        session = self._resolve_session_name()
        self.result_csv_file = os.path.join(out_dir, f"evaluation_results_{session}.csv")
        self.processed_images_file = os.path.join(out_dir, f"processed_images_{session}.txt")

        self.logger.info(f"Base directory: {self.base_directory}")

    def _load_processed_images(self) -> None:
        if self.processed_images_file and os.path.exists(self.processed_images_file):
            with open(self.processed_images_file, "r", encoding="utf-8") as f:
                self.processed_images = set(f.read().splitlines())
            self.logger.info(f"Loaded {len(self.processed_images)} previously processed images.")

        if self.result_csv_file and os.path.exists(self.result_csv_file):
            self.logger.info(f"Result file exists: {self.result_csv_file} — continuing from previous run.")

    # ============================================================
    # data loading
    # ============================================================
    def load_image_data(self) -> List[Dict[str, str]]:
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
            self.logger.info(f"Found {len(self.processed_images)} previously processed images. Resuming from there.")
        else:
            self.logger.info("No previously processed images found. Starting fresh.")

        if self.max_images is not None:
            self.logger.info(f"Max images this run: {self.max_images}")

        self.logger.info(f"Memory usage threshold set to {self.memory_threshold}% from config.")
        self.logger.info(f"Total images to process (raw): {self._total_images_to_process}")

    def load_data(self) -> List[Dict[str, str]]:
        raw_data = self.load_image_data()

        with self._processed_lock:
            todo = [d for d in raw_data if d.get("file_name") not in self.processed_images]

        return todo

    # ============================================================
    # batch processing
    # ============================================================
    def _process_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if self._stopping():
            return []

        if self._check_and_maybe_stop("batch_start"):
            return []

        if self._should_stop_by_max_images():
            self.logger.info(
                f"max_images limit reached before batch start "
                f"({self.processed_count_this_run}/{self.max_images}). stopping."
            )
            return []

        with self._processed_lock:
            batch = [img for img in batch if img.get("file_name") not in self.processed_images]

        self._dbg(f"_process_batch start: filtered_batch_size={len(batch)}")

        if not batch:
            self.logger.debug("All images in this batch are already processed. Skipping.")
            return []

        if self.max_images is not None or (self.max_workers or 1) == 1:
            results = self._process_batch_serial(batch)
        else:
            results = self._process_batch_parallel(batch)

        self._dbg(f"_process_batch after processing: results_count={len(results)}")

        if results and self.result_csv_file:
            self._dbg(f"before save_results: result_csv_file={self.result_csv_file}, " f"results_count={len(results)}")
            self.save_results(results, self.result_csv_file)
            self._dbg(f"after save_results: result_csv_file={self.result_csv_file}, " f"results_count={len(results)}")

        gc.collect()
        self.memory_monitor.log_usage(prefix="Post batch GC")
        self._check_and_maybe_stop("batch_end")

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
        self,
        file_name: str,
        orientation: str,
        bit_depth: str,
    ) -> Optional[Dict[str, Union[str, float, bool]]]:
        if self._stop_event.is_set():
            return None
        if self._check_and_maybe_stop("worker_start"):
            return None

        self.logger.info(f"Processing image: {file_name}")
        self._dbg(f"process_image start file={file_name}, orientation={orientation}, bit_depth={bit_depth}")

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

        image = None
        evaluator = None

        try:
            orientation_int: Optional[int]
            try:
                orientation_int = int(orientation) if orientation not in (None, "", "None") else None
            except Exception:
                orientation_int = None
                self.logger.warning(f"Invalid orientation value: {orientation} for {file_name}")

            self._dbg(
                f"before load_image: {file_name}, "
                f"output_bps=8, apply_exif_rotation=True, orientation={orientation_int}"
            )
            image = self.image_loader.load_image(
                filepath=file_name,
                output_bps=8,
                apply_exif_rotation=True,
                orientation=orientation_int,
            )
            self._dbg(
                f"after load_image: {file_name}, "
                f"shape={getattr(image, 'shape', None)}, dtype={getattr(image, 'dtype', None)}"
            )

            self._dbg(f"before evaluator init: {file_name}")
            evaluator = PortraitQualityEvaluator(
                image_input=image,
                is_raw=False,
                logger=self.logger,
                file_name=file_name,
                config_manager=self.config_manager,
                quality_profile=self.config.get("quality_profile", "portrait"),
                thresholds_path=self.config.get("evaluator_thresholds_path"),
            )
            self._dbg(f"after evaluator init: {file_name}")

            self._dbg(f"before evaluator.evaluate: {file_name}")
            eval_result = evaluator.evaluate()
            self._dbg(f"after evaluator.evaluate: {file_name}, " f"result_type={type(eval_result).__name__}")

            if eval_result:
                result.update(eval_result)
                return result

            self.logger.warning(f"No evaluation result for image {file_name}")

        except FileNotFoundError:
            self.logger.error(f"File not found: {file_name}")
        except Exception as e:
            self.logger.error(f"Error processing image {file_name}: {e}", exc_info=True)
            self._dbg(f"process_image exception: {file_name}, type={type(e).__name__}, repr={e!r}")
        finally:
            try:
                if evaluator is not None:
                    del evaluator
                if image is not None:
                    del image
            except Exception as e:
                self._dbg(f"process_image finally cleanup exception: {file_name}, {e!r}")

            gc.collect()

        return None

    def _process_single_image(self, img_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
        file_name = img_info.get("file_name")

        if self._stop_event.is_set():
            return None

        if self._should_stop_by_max_images():
            return None

        try:
            if not self.base_directory:
                raise RuntimeError("base_directory is not set")

            full_path = os.path.join(self.base_directory, img_info["file_name"])

            result = self.process_image(
                full_path,
                img_info["orientation"],
                img_info["bit_depth"],
            )

            if result:
                self.logger.info(f"Processed image: {img_info['file_name']}")

                self._dbg(f"before _mark_as_processed: {file_name}")
                self._mark_as_processed(img_info["file_name"])
                self._dbg(f"after _mark_as_processed: {file_name}")

                self.processed_count_this_run = int(self.processed_count_this_run or 0) + 1

                if self._should_stop_by_max_images():
                    self.logger.info(
                        f"max_images limit reached "
                        f"({self.processed_count_this_run}/{self.max_images}). stopping run."
                    )

                return result

        except Exception as e:
            self.logger.error(
                f"Failed to process image {img_info.get('file_name')}: {e}",
                exc_info=True,
            )
            self._dbg(f"_process_single_image exception: {file_name}, " f"type={type(e).__name__}, repr={e!r}")

        return None

    def _process_batch_serial(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for idx, img_info in enumerate(batch, start=1):
            file_name = img_info.get("file_name")
            self._dbg(f"_process_batch_serial loop start idx={idx}, file={file_name}")

            if self._stop_event.is_set():
                break
            if self._check_and_maybe_stop("serial_loop"):
                break
            if self._should_stop_by_max_images():
                break

            result = self._process_single_image(img_info)

            if result:
                results.append(result)

        return results

    def _process_batch_parallel(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers or 2) as executor:
            futures = []

            for idx, img_info in enumerate(batch, start=1):
                file_name = img_info.get("file_name")
                if self._stop_event.is_set():
                    break
                if self._check_and_maybe_stop("parallel_before_submit"):
                    break
                if self._should_stop_by_max_images():
                    break

                self._dbg(f"_process_batch_parallel submit idx={idx}, file={file_name}")
                futures.append(executor.submit(self._process_single_image, img_info))

            cancelled_pending = False
            for future_idx, future in enumerate(as_completed(futures), start=1):
                if self._stop_event.is_set() and not cancelled_pending:
                    cancelled_pending = True
                    for f in futures:
                        f.cancel()

                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except CancelledError:
                    self._dbg(f"_process_batch_parallel future cancelled idx={future_idx}")
                    continue
                except Exception as e:
                    self.logger.exception("[Parallel] Failed to process image", exc_info=True)
                    self._dbg(
                        f"_process_batch_parallel future exception idx={future_idx}, "
                        f"type={type(e).__name__}, repr={e!r}"
                    )
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

                csvfile.flush()
                os.fsync(csvfile.fileno())

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

        self._dbg(
            f"cleanup counts: processed_this_run={processed_this_run}, "
            f"remaining_count={remaining_count}, "
            f"total_images={self._total_images_to_process}, "
            f"start_processed={self._start_processed_count}"
        )

        self.processed_count_this_run = int(processed_this_run)

        if (
            self.max_images is not None
            and not getattr(self, "memory_threshold_exceeded", False)
            and int(self.processed_count_this_run or 0) > 0
            and int(self.processed_count_this_run or 0) < int(self._total_images_to_process or 0)
        ):
            if not getattr(self, "stop_reason", None):
                self.stop_reason = "max_images_limit"

        if getattr(self, "memory_threshold_exceeded", False):
            if not getattr(self, "stop_reason", None):
                self.stop_reason = "memory_threshold"

            self.logger.warning(
                f"Batch processing stopped due to memory threshold. "
                f"Processed {processed_this_run} images, {remaining_count} remaining. "
                "You can re-run to continue remaining images."
            )
        elif getattr(self, "stop_reason", None) == "max_images_limit":
            self.logger.info(
                f"Batch processing stopped by max_images. "
                f"Processed {processed_this_run} images, {remaining_count} remaining. "
                "You can re-run to continue remaining images."
            )
        elif getattr(self, "completed_all_batches", False):
            self.logger.info(f"All batches processed successfully. Total processed images: {processed_this_run}")

        processed_file = getattr(self, "processed_images_file", None)
        if (
            getattr(self, "completed_all_batches", False)
            and not getattr(self, "memory_threshold_exceeded", False)
            and getattr(self, "stop_reason", None) is None
            and processed_file
            and os.path.exists(processed_file)
        ):
            self.logger.info(f"Removing processed images file: {processed_file}")
            os.remove(processed_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process images with PortraitQualityBatchProcessor.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=("Config file path (optional). " "If omitted, ConfigManager uses CONFIG_ENV / defaults."),
    )
    parser.add_argument(
        "--config_env",
        type=str,
        default=None,
        help=("Config environment (e.g. prod/test). " "If omitted, CONFIG_ENV env-var may be used."),
    )
    parser.add_argument(
        "--config_paths",
        nargs="*",
        default=None,
        help="Optional explicit config file list (applied in order; supports extends).",
    )
    parser.add_argument("--date", type=str, help="Specify the date for directory and file names.")
    parser.add_argument("--max_workers", type=int, help="Number of worker threads")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Max images to process in this run",
    )
    parser.add_argument(
        "--input_csv_path",
        type=str,
        default=None,
        help="Explicit input CSV path generated by a previous NEF stage.",
    )
    args = parser.parse_args()
    print("[DEBUG] args:", args)

    processor = PortraitQualityBatchProcessor(
        config_path=args.config_path,
        config_env=args.config_env,
        config_paths=args.config_paths,
        date=args.date,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_images=args.max_images,
        input_csv_path=args.input_csv_path,
    )
    processor.execute()
