from __future__ import annotations

import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor
from photo_insight.adapters.lightroom.xmp import (
    create_new_xmp,
    merge_into_existing_xmp,
)
from photo_insight.pipelines.xmp_export._internal.csv_contract import (
    compute_pick_from_csv,
    get_str,
    normalize_lr_color_label,
    normalize_lr_label_key,
    safe_float,
    safe_int,
    safe_int_flag,
)
from photo_insight.pipelines.xmp_export._internal.file_locator import (
    build_nef_index,
    find_csv,
    resolve_image_root,
)


def _resolve_xmp_values_from_row(
    row: dict[str, Any],
    *,
    pick_mode: str,
) -> dict[str, Any]:
    overall = safe_float(row.get("overall_score"), 0.0)

    lr_rating = safe_int(row.get("lr_rating", -1), default=-1)
    rating = lr_rating if lr_rating >= 0 else safe_int(overall // 20, default=0)

    lr_label_key = normalize_lr_label_key(get_str(row, "lr_labelcolor_key", ""))
    lr_label_display = get_str(row, "lr_label_display", "")

    if not lr_label_key:
        lr_color_label = get_str(row, "lr_color_label", "")
        lr_label_key, inferred_display = normalize_lr_color_label(lr_color_label)
        if not lr_label_display:
            lr_label_display = inferred_display or ""

    label_key = lr_label_key if lr_label_key else None
    label_display = lr_label_display if lr_label_display else None

    accepted_flag = safe_int_flag(row.get("accepted_flag", 0))
    secondary_flag = safe_int_flag(row.get("secondary_accept_flag", 0))
    top_flag = safe_int_flag(row.get("flag", 0))

    pick = compute_pick_from_csv(
        pick_mode=pick_mode,
        accepted_flag=accepted_flag,
        secondary_flag=secondary_flag,
        top_flag=top_flag,
    )

    lr_keywords = get_str(row, "lr_keywords", "").strip()
    keywords = [lr_keywords] if lr_keywords else None

    return {
        "rating": rating,
        "pick": pick,
        "label_key": label_key,
        "label_display": label_display,
        "keywords": keywords,
    }


def _apply_row_to_xmp(
    *,
    nef_name: str,
    nef_path: Path,
    dry_run: bool,
    backup_xmp: bool,
    force_rating: bool,
    force_pick: bool,
    force_color: bool,
    clear_color_if_pick0: bool,
    write_keywords: bool,
    overwrite_keywords: bool,
    resolved: dict[str, Any],
) -> str:
    xmp_path = nef_path.with_suffix(".xmp")

    rating = resolved["rating"]
    pick = resolved["pick"]
    label_key = resolved["label_key"]
    label_display = resolved["label_display"]
    keywords = resolved["keywords"]

    if xmp_path.exists():
        if backup_xmp and not dry_run:
            shutil.copy(xmp_path, xmp_path.with_suffix(".xmp.bak"))

        merge_into_existing_xmp(
            xmp_path,
            rating,
            pick,
            label_key,
            label_display,
            keywords=keywords,
            dry_run=dry_run,
            force_rating=force_rating,
            force_pick=force_pick,
            force_color=force_color,
            clear_color_if_pick0=clear_color_if_pick0,
            write_keywords=write_keywords,
            overwrite_keywords=overwrite_keywords,
        )

        print(
            f"🔁 MERGE {nef_name} ★{rating} Pick={pick} "
            f"Color={label_display or ''} "
            f"KW={'Y' if (write_keywords and keywords) else 'N'}"
        )
        return "merged"

    if dry_run:
        print(
            f"[DRY] NEW {nef_name} ★{rating} Pick={pick} "
            f"Color={label_display or ''} "
            f"KW={'Y' if (write_keywords and keywords) else 'N'}"
        )
        return "created"

    xmp = create_new_xmp(
        rating,
        pick,
        label_key,
        label_display,
        keywords=keywords if write_keywords else None,
    )
    ET.ElementTree(xmp).write(xmp_path, encoding="utf-8", xml_declaration=True)

    print(
        f"✨ NEW   {nef_name} ★{rating} Pick={pick} "
        f"Color={label_display or ''} "
        f"KW={'Y' if (write_keywords and keywords) else 'N'}"
    )
    return "created"


class XmpExportProcessor(BaseBatchProcessor):
    """
    evaluation_rank の CSV を入力として、
    Lightroom XMP を生成・更新する stage。
    """

    runtime_param_names = ("date", "target_dir", "input_csv_path", "max_images")

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_workers: int = 1,
        date: Optional[str] = None,
        target_dir: Optional[str] = None,
        input_csv_path: Optional[str] = None,
        config_env: Optional[str] = None,
        config_paths: Optional[list[str]] = None,
        resolver: Any = None,
        loader: Any = None,
        watch_factory: Any = None,
        list_policy: str = "replace",
        strict_missing: bool = True,
        auto_load: bool = True,
    ) -> None:
        super().__init__(
            config_path=config_path,
            config_env=config_env,
            config_paths=config_paths,
            max_workers=max_workers,
            logger=None,
            resolver=resolver,
            loader=loader,
            watch_factory=watch_factory,
            list_policy=list_policy,
            strict_missing=strict_missing,
            auto_load=auto_load,
        )

        self.date = date
        self.target_dir = target_dir
        self.input_csv_path = input_csv_path

        self.csv_path: Optional[Path] = None
        self.resolved_image_root: Optional[Path] = None
        self.nef_index: dict[str, Path] = {}

        self.backup_xmp = True
        self.force_rating = False
        self.force_pick = False
        self.force_color = False
        self.clear_color_if_pick0 = False
        self.pick_mode = "flags"
        self.write_keywords = False
        self.overwrite_keywords = False

    def setup(self) -> None:
        xmp_cfg = self.config_manager.get_config().get("xmp_export", {}) if self.config_manager.get_config() else {}

        output_dir = Path(xmp_cfg.get("output_dir", "output"))
        csv_glob = xmp_cfg.get("csv_glob", "evaluation_ranking_*.csv")
        image_root = Path(xmp_cfg.get("image_root", "/work/input"))

        self.backup_xmp = bool(xmp_cfg.get("backup_xmp", True))
        self.force_rating = bool(xmp_cfg.get("force_rating", False))
        self.force_pick = bool(xmp_cfg.get("force_pick", False))
        self.force_color = bool(xmp_cfg.get("force_color", False))
        self.clear_color_if_pick0 = bool(xmp_cfg.get("clear_color_if_pick0", False))
        self.pick_mode = xmp_cfg.get("pick_mode", "flags")
        self.write_keywords = bool(xmp_cfg.get("write_keywords", False))
        self.overwrite_keywords = bool(xmp_cfg.get("overwrite_keywords", False))

        if self.input_csv_path:
            self.csv_path = Path(self.input_csv_path)
        else:
            self.csv_path = find_csv(output_dir, csv_glob, self.date)

        if self.target_dir:
            self.resolved_image_root = Path(self.target_dir)
        else:
            self.resolved_image_root = resolve_image_root(image_root, self.date)

        self.nef_index = build_nef_index(self.resolved_image_root)

        self.logger.info(f"Using CSV: {self.csv_path}")
        self.logger.info(f"Resolved image root: {self.resolved_image_root}")
        self.logger.info(f"Indexed NEF files: {len(self.nef_index)}")

        super().setup()

    def load_data(self) -> list[dict[str, Any]]:
        if self.csv_path is None:
            raise RuntimeError("csv_path is not resolved")

        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)

        with self.csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _process_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        dry_run = bool(getattr(self, "dry_run", False))

        for row in batch:
            nef_name = get_str(row, "file_name")
            if not nef_name:
                print("⚠️ file_name missing, skip row")
                results.append(
                    {
                        "status": "failure",
                        "score": 0.0,
                        "row": row,
                        "error": "file_name missing",
                    }
                )
                continue

            nef_path = self.nef_index.get(nef_name)
            if nef_path is None or not nef_path.exists():
                print(f"❌ NEF not found: {nef_name}")
                results.append(
                    {
                        "status": "failure",
                        "score": 0.0,
                        "row": row,
                        "error": f"NEF not found: {nef_name}",
                    }
                )
                continue

            try:
                resolved = _resolve_xmp_values_from_row(row, pick_mode=self.pick_mode)
                action = _apply_row_to_xmp(
                    nef_name=nef_name,
                    nef_path=nef_path,
                    dry_run=dry_run,
                    backup_xmp=self.backup_xmp,
                    force_rating=self.force_rating,
                    force_pick=self.force_pick,
                    force_color=self.force_color,
                    clear_color_if_pick0=self.clear_color_if_pick0,
                    write_keywords=self.write_keywords,
                    overwrite_keywords=self.overwrite_keywords,
                    resolved=resolved,
                )

                out_row = dict(row)
                out_row["xmp_export_action"] = action
                results.append(
                    {
                        "status": "success",
                        "score": float(resolved["rating"]),
                        "row": out_row,
                    }
                )
            except Exception as e:
                self.logger.exception("XMP export failed")
                results.append(
                    {
                        "status": "failure",
                        "score": 0.0,
                        "row": row,
                        "error": str(e),
                    }
                )

        return results

    def cleanup(self) -> None:
        try:
            success = 0
            failure = 0
            for r in getattr(self, "all_results", []):
                if r.get("status") == "success":
                    success += 1
                else:
                    failure += 1

            self.logger.info(
                f"[xmp_export] completed success={success} failure={failure} "
                f"csv_path={self.csv_path} image_root={self.resolved_image_root}"
            )
        finally:
            super().cleanup()
