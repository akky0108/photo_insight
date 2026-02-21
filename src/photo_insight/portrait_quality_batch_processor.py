# src/portrait_quality_batch_processor.py
from __future__ import annotations

import os
import csv
import gc
from typing import Any, List, Dict, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        """
        バッチ処理クラスの初期化。
        設定ファイルやロガー、日付指定などを受け取る。
        """
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

        # Base が持つ logger/config_manager を上書きしない方針に寄せる。
        # ただし互換のため、外部 logger が渡された場合はそれを採用。
        self.config = self.config_manager.get_config()
        if logger is not None:
            self.logger = logger
        else:
            self.logger = self.config_manager.get_logger(
                "PortraitQualityBatchProcessor"
            )

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

        # Base が completed_all_batches を持つが、ここでも互換で保持
        self.completed_all_batches = False
        self.memory_threshold_exceeded = False

        self.batch_size = batch_size or self.config.get("batch_size", 10)
        self.memory_threshold = self.config_manager.get_memory_threshold(default=90)

        # run bookkeeping（運用ログ用）
        self._start_processed_count: int = 0
        self._total_images_to_process: int = 0

    def on_config_change(self, new_config: dict) -> None:
        """
        設定ファイルが変更された際のハンドラ。
        バッチサイズや出力先ディレクトリを更新する。
        """
        self.logger.info("Config updated. Applying new settings...")
        self.batch_size = new_config.get("batch_size", self.batch_size)

        out_dir = new_config.get("output_directory") or self.output_directory or "temp"
        self.output_directory = out_dir
        os.makedirs(self.output_directory, exist_ok=True)

    def setup(self) -> None:
        """
        バッチ処理の事前準備。
        ディレクトリやファイルパスの設定、既処理データの読み込みなどを行う。
        """
        self.logger.info("Setting up PortraitQualityBatchProcessor...")
        self._set_directories_and_files()
        self._load_processed_images()

        # ★開始時点の処理済み数（累計）を保存して差分で数える
        self._start_processed_count = len(self.processed_images)

        # Base契約: setup() -> self.data = self.get_data() -> after_data_loaded(self.data)
        super().setup()

        self.memory_threshold_exceeded = False
        self.completed_all_batches = False

    def _set_directories_and_files(self) -> None:
        """
        ディレクトリとファイルパスを設定するヘルパー。
        日付指定がある場合はそれに応じたパスを生成。
        """
        import re

        picture_root = self.config.get("picture_root", "/mnt/l/picture")

        self.output_directory = self.config.get("output_directory", "temp")
        os.makedirs(self.output_directory, exist_ok=True)

        if self.date:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.date):
                raise ValueError(
                    f"Invalid date format: {self.date} (expected YYYY-MM-DD)"
                )
            year = self.date[:4]
            self.base_directory = os.path.join(picture_root, year, self.date)
        else:
            self.base_directory = self.config.get(
                "base_directory", "/mnt/l/picture/2025/2025-01-01"
            )

        if self.date:
            self.image_csv_file = os.path.join(
                self.output_directory, f"{self.date}_raw_exif_data.csv"
            )
            self.result_csv_file = os.path.join(
                self.output_directory, f"evaluation_results_{self.date}.csv"
            )
            self.processed_images_file = os.path.join(
                self.output_directory, f"processed_images_{self.date}.txt"
            )
        else:
            self.image_csv_file = os.path.join(
                self.output_directory, "nef_exif_data.csv"
            )
            self.result_csv_file = os.path.join(
                self.output_directory, "evaluation_results.csv"
            )
            self.processed_images_file = os.path.join(
                self.output_directory, "processed_images.txt"
            )

        # 入力元は作らない。存在しないなら即落として設定ミスを早期発見する。
        if not self.base_directory or not os.path.isdir(self.base_directory):
            raise FileNotFoundError(
                f"Base directory does not exist: {self.base_directory}"
            )

        self.logger.info(f"ベースディレクトリ: {self.base_directory}")

    def _load_processed_images(self) -> None:
        """
        既に処理済みの画像ファイル名を読み込む。
        """
        if self.processed_images_file and os.path.exists(self.processed_images_file):
            with open(self.processed_images_file, "r", encoding="utf-8") as f:
                self.processed_images = set(f.read().splitlines())
            self.logger.info(
                f"Loaded {len(self.processed_images)} previously processed images."
            )

        if self.result_csv_file and os.path.exists(self.result_csv_file):
            self.logger.info(
                f"Result file exists: {self.result_csv_file} — "
                f"continuing from previous run."
            )

    def load_image_data(self) -> List[Dict[str, str]]:
        """
        画像のメタデータ（ファイル名、向き、ビット深度）を CSV から読み込む。

        Returns:
            List[Dict[str, str]]: 画像情報のリスト
        """
        if not self.image_csv_file:
            self.logger.warning(
                "image_csv_file is not set. Proceeding with empty data."
            )
            return []

        self.logger.info(f"Loading image data from {self.image_csv_file}")
        image_data: List[Dict[str, str]] = []
        try:
            with open(self.image_csv_file, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                required_keys = {"FileName", "Orientation", "BitDepth"}

                if not required_keys.issubset(set(reader.fieldnames or [])):
                    self.logger.error(
                        f"Missing required columns in CSV. Found: {reader.fieldnames}"
                    )
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
            self.logger.warning(
                f"File not found: {self.image_csv_file}. Proceeding with empty data."
            )
        except csv.Error as e:
            self.logger.error(f"CSV parsing error in {self.image_csv_file}: {e}")
        return image_data

    def after_data_loaded(self, data: List[Dict]) -> None:
        """
        データロード後に一度だけ行う副作用をここへ集約（Base.setup() から呼ばれる）。
        - 処理済みの存在ログ
        - メモリ閾値ログ
        - 対象件数ログ（未処理のみ data に入る想定）
        """
        self._total_images_to_process = len(data)
        if self.processed_images:
            self.logger.info(
                f"Found {len(self.processed_images)} previously processed images. "
                f"Resuming from there."
            )
        else:
            self.logger.info("No previously processed images found. Starting fresh.")

        self.logger.info(
            f"Memory usage threshold set to {self.memory_threshold}% from config."
        )
        self.logger.info(f"Total images to process: {self._total_images_to_process}")

    def _process_batch(self, batch: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        指定された画像バッチを処理する。
        処理済み画像はスキップし、結果を CSV に保存する。
        バッチ終了後にメモリ使用量をチェックし、閾値超過なら中断フラグを立てる。
        """
        # 未処理画像のみ残す
        with self._processed_lock:
            batch = [
                img
                for img in batch
                if img.get("file_name") not in self.processed_images
            ]

        if not batch:
            self.logger.debug(
                "All images in this batch are already processed. Skipping."
            )
            return []

        # 並列／直列処理
        if (self.max_workers or 1) == 1:
            results = self._process_batch_serial(batch)
        else:
            results = self._process_batch_parallel(batch)

        # 結果を保存
        if results and self.result_csv_file:
            self.save_results(results, self.result_csv_file)

        # ガベージコレクションとメモリログ
        gc.collect()
        self.memory_monitor.log_usage(prefix="Post batch GC")

        # メモリ閾値チェック
        mem_usage = self.memory_monitor.get_memory_usage()
        if mem_usage > self.memory_threshold:
            self.logger.warning(
                f"Memory usage {mem_usage:.1f}% "
                f"exceeded threshold ({self.memory_threshold}%). "
                f"Will stop after this batch."
            )
            self.memory_threshold_exceeded = True

        return results or []

    def process_image(
        self, file_name: str, orientation: str, bit_depth: str
    ) -> Optional[Dict[str, Union[str, float, bool]]]:
        """
        単一画像を読み込み、評価処理を行い、結果を辞書形式で返す。

        Args:
            file_name (str): 画像ファイルパス
            orientation (str): 画像の向き
            bit_depth (str): ビット深度

        Returns:
            Optional[Dict[str, Union[str, float, bool]]]: 評価結果または None
        """
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

            # 明示解放（大きい配列を抱える場合の保険）
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

    def _process_single_image(
        self, img_info: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
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

    def _process_batch_serial(
        self, batch: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for img_info in batch:
            result = self._process_single_image(img_info)
            if result:
                results.append(result)
        return results

    def _process_batch_parallel(
        self, batch: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers or 2) as executor:
            future_to_img = {
                executor.submit(self._process_single_image, img_info): img_info
                for img_info in batch
            }

            for future in as_completed(future_to_img):
                img_info = future_to_img[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception:
                    # 1枚の例外でバッチ全体が落ちないようにする
                    self.logger.exception(
                        f"[Parallel] Failed to process image "
                        f"{img_info.get('file_name')}",
                        exc_info=True,
                    )
                    continue

        return results

    def _mark_as_processed(self, file_name: str) -> None:
        """
        処理済み画像として記録し、再処理を防止する。
        """
        if not self.processed_images_file:
            raise RuntimeError("processed_images_file is not set")

        # ファイル追記と set 更新を同一ロックで保護
        with self._processed_lock:
            if file_name in self.processed_images:
                return

            with open(self.processed_images_file, "a", encoding="utf-8") as f:
                f.write(f"{file_name}\n")
                f.flush()
                os.fsync(f.fileno())  # 再開前提では有効

            self.processed_images.add(file_name)
            self.logger.debug(f"Marked as processed: {file_name}")

    def save_results(
        self, results: List[Dict[str, Union[str, float, bool]]], file_path: str
    ) -> None:
        """
        画像評価の結果を CSV ファイルに保存する。
        """
        self.logger.info(f"Saving results to {file_path}")
        file_exists = os.path.isfile(file_path)
        results_sorted = sorted(results, key=lambda x: str(x.get("file_name", "")))

        header_generator = PortraitQualityHeaderGenerator()
        fieldnames = header_generator.get_all_headers()

        with self.get_lock():
            with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, extrasaction="ignore"
                )
                if not file_exists:
                    writer.writeheader()

                for row in results_sorted:
                    writer.writerow({key: row.get(key, None) for key in fieldnames})

    def cleanup(self) -> None:
        """
        後処理のクリーンアップ処理。
        全件処理が完了していれば、一時ファイルを削除する。
        """
        super().cleanup()
        self.logger.info(
            "Cleaning up PortraitQualityBatchProcessor-specific resources."
        )

        processed_this_run = len(self.processed_images) - int(
            self._start_processed_count or 0
        )
        remaining_count = int(self._total_images_to_process or 0) - processed_this_run
        if remaining_count < 0:
            remaining_count = 0

        if getattr(self, "memory_threshold_exceeded", False):
            self.logger.warning(
                f"Batch processing stopped due to memory threshold. "
                f"Processed {processed_this_run} images, {remaining_count} remaining. "
                "You can re-run to continue remaining images."
            )
        elif getattr(self, "completed_all_batches", False):
            self.logger.info(
                f"All batches processed successfully. "
                f"Total processed images: {processed_this_run}"
            )

        processed_file = getattr(self, "processed_images_file", None)
        if (
            getattr(self, "completed_all_batches", False)
            and processed_file
            and os.path.exists(processed_file)
        ):
            self.logger.info(f"Removing processed images file: {processed_file}")
            os.remove(processed_file)

    def load_data(self) -> List[Dict[str, str]]:
        """
        BaseBatchProcessor の新契約:
        - load_data(): 純I/O（副作用なし）
        - キャッシュは Base が握る（get_data() は Base 側）

        Returns:
            List[Dict[str, str]]: 画像メタデータ一覧（未処理のものだけ）
        """
        raw_data = self.load_image_data()
        with self._processed_lock:
            return [
                d for d in raw_data if d.get("file_name") not in self.processed_images
            ]


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
        help=(
            "Optional explicit config file list "
            "(applied in order; supports extends)."
        ),
    )
    parser.add_argument(
        "--date", type=str, help="Specify the date for directory and file names."
    )
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
