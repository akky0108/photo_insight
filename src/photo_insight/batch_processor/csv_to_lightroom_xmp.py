#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV → Lightroom XMP batch (photo_insight evaluation_rank outputs)

方針（運用をきれいにする）:
- 「評価ランクCSVが決めた lr_* を信頼して XMP に反映」する
  - lr_rating / lr_labelcolor_key / lr_label_display / lr_color_label / lr_keywords を優先
- このスクリプト側で rating/color を二重計算しない（矛盾を作らない）
- pick は運用上の利便のため、CSVの採用系フラグ（accepted / secondary / top_flag）から決める（既定）
- CSV由来の "True"/"False"/"1"/"0" 事故を回避する safe_* を採用

オプション:
- --write-keywords で lr_keywords を XMP キーワード(dc:subject)へ追記（既存を尊重する既定ポリシー）
"""

from __future__ import annotations

import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Any, List

# =========================================================
# 設定（CLIで上書き可能）
# =========================================================

OUTPUT_DIR = Path("output")
CSV_GLOB = "evaluation_ranking_*.csv"

DRY_RUN = False
BACKUP_XMP = True

BASE_DIRECTORY_ROOT = Path("/mnt/l/picture/")

# Pick 判定（運用きれい化：既定はCSVの採用/候補フラグ由来）
# - accepted_flag == 1 → Pick
# - secondary_accept_flag == 1 → Pick
# - flag == 1（top candidate） → Pick
DEFAULT_PICK_MODE = "flags"  # flags|accepted|accepted_or_secondary|none

# =========================================================
# XML Namespace
# =========================================================

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "xmpDM": "http://ns.adobe.com/xmp/1.0/DynamicMedia/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

ET.register_namespace("x", NS["x"])
ET.register_namespace("rdf", NS["rdf"])
ET.register_namespace("xmp", NS["xmp"])
ET.register_namespace("xmpDM", NS["xmpDM"])
ET.register_namespace("photoshop", NS["photoshop"])
ET.register_namespace("lr", NS["lr"])
ET.register_namespace("dc", NS["dc"])


# =========================================================
# CSV → XMP 用ユーティリティ（安全系）
# =========================================================


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return False


def safe_int(value: Any, default: int = 0) -> int:
    """一般 int（True/False も許容）"""
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    s = str(value).strip()
    try:
        return int(float(s))
    except Exception:
        return default


def safe_int_flag(value: Any, default: int = 0) -> int:
    """
    CSV由来の 0/1, True/False, "TRUE"/"False" を 0/1 に正規化する。
    int("False") 事故を確実に回避する。
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0

    s = str(value).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return 1
    if s in ("0", "false", "f", "no", "n"):
        return 0

    try:
        return 1 if int(float(s)) != 0 else 0
    except Exception:
        return default


def get_str(row: dict, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return str(value)


# =========================================================
# CSV最新版: lr_color_label/lr_labelcolor_key 対応
# =========================================================

COLOR_LABEL_MAP = {
    "green": ("green", "グリーン"),
    "yellow": ("yellow", "イエロー"),
    "blue": ("blue", "ブルー"),
    "red": ("red", "レッド"),
    "purple": ("purple", "パープル"),
    "none": (None, None),
    "": (None, None),
}


def normalize_lr_color_label(
    lr_color_label: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    lr_color_label ("Green"/"green") から key/display を推定（フォールバック用）
    """
    if not lr_color_label:
        return None, None
    key = lr_color_label.strip().lower()
    return COLOR_LABEL_MAP.get(key, (key, None))


def normalize_lr_label_key(key: str) -> Optional[str]:
    if not key:
        return None
    k = key.strip().lower()
    return k if k in {"red", "yellow", "green", "blue", "purple"} else None


# =========================================================
# ファイル探索ユーティリティ
# =========================================================


def find_csv(output_dir: Path, csv_glob: str, date: Optional[str]) -> Path:
    """
    dateがあるなら evaluation_ranking_{date}.csv を優先。
    無ければ glob の最新版。
    """
    if date:
        candidate = output_dir / f"evaluation_ranking_{date}.csv"
        if candidate.exists():
            return candidate

    csv_files = sorted(
        output_dir.glob(csv_glob),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV found: {output_dir}/{csv_glob}")
    return csv_files[0]


def resolve_image_root(image_root: Path, date: Optional[str]) -> Path:
    """
    dateが指定されたら /root/YYYY/YYYY-MM-DD に固定する。
    """
    if not date:
        return image_root
    year = date.split("-")[0]
    return image_root / year / date


def build_nef_index(base_dir: Path) -> dict[str, Path]:
    """
    base_dir 配下の *.NEF を index化（同名衝突は検知して警告）
    """
    index: dict[str, Path] = {}
    dup: dict[str, list[Path]] = {}

    for p in base_dir.rglob("*.NEF"):
        name = p.name
        if name in index:
            dup.setdefault(name, [index[name]]).append(p)
            # 後勝ちにせず「最初のを維持」して安全に倒す
            continue
        index[name] = p

    if dup:
        sample = list(dup.items())[:5]
        print(
            "⚠️ Duplicate NEF names detected under the search root. "
            "(showing up to 5)"
        )
        for name, paths in sample:
            print(f"  - {name}:")
            for pp in paths:
                print(f"      {pp}")

    return index


# =========================================================
# Pick の決め方（運用きれい化）
# =========================================================


def compute_pick_from_csv(
    *,
    pick_mode: str,
    accepted_flag: int,
    secondary_flag: int,
    top_flag: int,
) -> int:
    """
    pick_mode:
      - flags: accepted or secondary or top_flag
      - accepted: accepted only
      - accepted_or_secondary: accepted or secondary
      - none: always 0
    """
    m = (pick_mode or "").strip().lower()
    if m == "none":
        return 0
    if m == "accepted":
        return 1 if accepted_flag == 1 else 0
    if m == "accepted_or_secondary":
        return 1 if (accepted_flag == 1 or secondary_flag == 1) else 0
    # default: flags
    return 1 if (accepted_flag == 1 or secondary_flag == 1 or top_flag == 1) else 0


# =========================================================
# XMP 操作ユーティリティ
# =========================================================


def find_target_description(root: ET.Element) -> Optional[ET.Element]:
    for desc in root.findall(".//rdf:Description", NS):
        # 互換のため「それっぽい」子要素がある Description を優先
        for child in desc:
            if (
                child.tag.startswith(f"{{{NS['xmp']}}}")
                or child.tag.startswith(f"{{{NS['lr']}}}")
                or child.tag.startswith(f"{{{NS['xmpDM']}}}")
                or child.tag.startswith(f"{{{NS['photoshop']}}}")
                or child.tag.startswith(f"{{{NS['dc']}}}")
            ):
                return desc
        return desc
    return root.find(".//rdf:Description", NS)


def create_new_xmp(
    rating: int,
    pick: int,
    label_key: Optional[str],
    label_display: Optional[str],
    *,
    keywords: Optional[List[str]] = None,
) -> ET.Element:
    xmpmeta = ET.Element(f"{{{NS['x']}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, f"{{{NS['rdf']}}}RDF")
    desc = ET.SubElement(
        rdf,
        f"{{{NS['rdf']}}}Description",
        attrib={f"{{{NS['rdf']}}}about": ""},
    )

    desc.set(f"{{{NS['xmp']}}}Rating", str(int(rating)))
    desc.set(f"{{{NS['xmpDM']}}}pick", str(int(pick)))

    if label_key:
        desc.set(f"{{{NS['photoshop']}}}LabelColor", label_key)
    if label_display:
        desc.set(f"{{{NS['xmp']}}}Label", label_display)

    if keywords:
        _ensure_dc_subject(desc, keywords, overwrite=False)

    return xmpmeta


def _clear_color_attrs(desc: ET.Element) -> None:
    k = f"{{{NS['photoshop']}}}LabelColor"
    label_tag = f"{{{NS['xmp']}}}Label"
    if k in desc.attrib:
        del desc.attrib[k]
    if label_tag in desc.attrib:
        del desc.attrib[label_tag]


def _get_or_create_bag(desc: ET.Element) -> ET.Element:
    """
    dc:subject は通常:
      <dc:subject>
        <rdf:Bag>
          <rdf:li>keyword</rdf:li>
        </rdf:Bag>
      </dc:subject>
    """
    subject = desc.find(f"{{{NS['dc']}}}subject")
    if subject is None:
        subject = ET.SubElement(desc, f"{{{NS['dc']}}}subject")
    bag = subject.find(f"{{{NS['rdf']}}}Bag")
    if bag is None:
        bag = ET.SubElement(subject, f"{{{NS['rdf']}}}Bag")
    return bag


def _existing_keywords(desc: ET.Element) -> List[str]:
    subject = desc.find(f"{{{NS['dc']}}}subject")
    if subject is None:
        return []
    bag = subject.find(f"{{{NS['rdf']}}}Bag")
    if bag is None:
        return []
    out: List[str] = []
    for li in bag.findall(f"{{{NS['rdf']}}}li"):
        if li.text:
            out.append(li.text.strip())
    return out


def _ensure_dc_subject(
    desc: ET.Element, keywords: List[str], *, overwrite: bool
) -> None:
    """
    keywords を dc:subject に反映。
    overwrite=False の場合は既存キーワードを尊重しつつ「無ければ追加」。
    """
    kw = [k.strip() for k in keywords if k and str(k).strip()]
    if not kw:
        return

    if overwrite:
        # 既存の dc:subject を作り直し
        subject = desc.find(f"{{{NS['dc']}}}subject")
        if subject is not None:
            desc.remove(subject)

    existing = set(_existing_keywords(desc)) if not overwrite else set()
    bag = _get_or_create_bag(desc)

    for k in kw:
        if k in existing:
            continue
        li = ET.SubElement(bag, f"{{{NS['rdf']}}}li")
        li.text = k
        existing.add(k)


def merge_into_existing_xmp(
    xmp_path: Path,
    rating: int,
    pick: int,
    label_key: Optional[str],
    label_display: Optional[str],
    *,
    keywords: Optional[List[str]],
    dry_run: bool,
    force_rating: bool,
    force_pick: bool,
    force_color: bool,
    clear_color_if_pick0: bool,
    write_keywords: bool,
    overwrite_keywords: bool,
):
    tree = ET.parse(xmp_path)
    root = tree.getroot()

    desc = find_target_description(root)
    if desc is None:
        raise RuntimeError("rdf:Description not found in XMP")

    # 1) ★ Rating：force_ratingがTrueなら上書き
    if force_rating:
        desc.set(f"{{{NS['xmp']}}}Rating", str(int(rating)))

    # 2) Pick：未設定(0/空)のみ or 強制
    existing_pick = (desc.get(f"{{{NS['xmpDM']}}}pick") or "").strip()
    if force_pick or existing_pick in ("", "0"):
        desc.set(f"{{{NS['xmpDM']}}}pick", str(int(pick)))

    # 3) Color：人が付けた色（既存）を守るのが基本
    existing_label = (desc.get(f"{{{NS['xmp']}}}Label") or "").strip()
    existing_key = (desc.get(f"{{{NS['photoshop']}}}LabelColor") or "").strip()
    has_existing_color = bool(existing_label or existing_key)

    if force_color:
        if pick == 0 and clear_color_if_pick0:
            _clear_color_attrs(desc)
        else:
            # label_key/display が None の場合は「何もしない」(既存保持)
            if label_key:
                desc.set(f"{{{NS['photoshop']}}}LabelColor", label_key)
            if label_display:
                desc.set(f"{{{NS['xmp']}}}Label", label_display)
    else:
        if not has_existing_color:
            if label_key:
                desc.set(f"{{{NS['photoshop']}}}LabelColor", label_key)
            if label_display:
                desc.set(f"{{{NS['xmp']}}}Label", label_display)

    # 4) Keywords：任意。既定は「既存尊重で追記」
    if write_keywords and keywords:
        _ensure_dc_subject(desc, keywords, overwrite=overwrite_keywords)

    # お掃除：lr:Pick / lr:ColorLabel があれば消す（混乱源）
    for tag in (f"{{{NS['lr']}}}Pick", f"{{{NS['lr']}}}ColorLabel"):
        node = desc.find(tag)
        if node is not None:
            desc.remove(node)

    if not dry_run:
        tree.write(xmp_path, encoding="utf-8", xml_declaration=True)


# =========================================================
# メイン処理
# =========================================================


def process_csv(
    csv_path: Path,
    nef_index: dict[str, Path],
    *,
    dry_run: bool,
    backup_xmp: bool,
    force_rating: bool,
    force_pick: bool,
    force_color: bool,
    clear_color_if_pick0: bool,
    pick_mode: str,
    write_keywords: bool,
    overwrite_keywords: bool,
):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            nef_name = get_str(row, "file_name")
            if not nef_name:
                print("⚠️ file_name missing, skip row")
                continue

            nef_path = nef_index.get(nef_name)
            if nef_path is None or not nef_path.exists():
                print(f"❌ NEF not found: {nef_name}")
                continue

            xmp_path = nef_path.with_suffix(".xmp")

            # -------------------------
            # CSVの lr_* を優先して採用
            # -------------------------
            overall = safe_float(row.get("overall_score"), 0.0)

            # Rating：lr_rating を信頼（無ければ -1 扱い）
            lr_rating = safe_int(row.get("lr_rating", -1), default=-1)
            rating = (
                lr_rating if lr_rating >= 0 else safe_int(overall // 20, default=0)
            )  # フォールバックは雑でOK

            # Color：lr_labelcolor_key/display を最優先。無ければ lr_color_label から推定。
            lr_label_key = normalize_lr_label_key(get_str(row, "lr_labelcolor_key", ""))
            lr_label_disp = get_str(row, "lr_label_display", "")

            if not lr_label_key:
                # フォールバック：lr_color_label から推定
                lr_color_label = get_str(row, "lr_color_label", "")
                lr_label_key, inferred_disp = normalize_lr_color_label(lr_color_label)
                if not lr_label_disp:
                    lr_label_disp = inferred_disp or ""

            # CSV 側が空のときは「自動では色を付けない」= None に倒す
            label_key = lr_label_key if lr_label_key else None
            label_display = lr_label_disp if lr_label_disp else None

            # Pick：運用を綺麗にするため CSVの採用/候補フラグで決める（既定）
            accepted_flag = safe_int_flag(row.get("accepted_flag", 0))
            secondary_flag = safe_int_flag(row.get("secondary_accept_flag", 0))
            top_flag = safe_int_flag(row.get("flag", 0))
            pick = compute_pick_from_csv(
                pick_mode=pick_mode,
                accepted_flag=accepted_flag,
                secondary_flag=secondary_flag,
                top_flag=top_flag,
            )

            # Keywords：lr_keywords をそのまま使う（短文化済み前提）
            lr_keywords = get_str(row, "lr_keywords", "").strip()
            keywords = [lr_keywords] if lr_keywords else None

            # -------------------------
            # XMP反映
            # -------------------------
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
                    f"KW={'Y' if (write_keywords and lr_keywords) else 'N'}"
                )
            else:
                if dry_run:
                    print(
                        f"[DRY] NEW {nef_name} ★{rating} Pick={pick} "
                        f"Color={label_display or ''} "
                        f"KW={'Y' if (write_keywords and lr_keywords) else 'N'}"
                    )
                    continue

                xmp = create_new_xmp(
                    rating,
                    pick,
                    label_key,
                    label_display,
                    keywords=keywords if write_keywords else None,
                )
                ET.ElementTree(xmp).write(
                    xmp_path, encoding="utf-8", xml_declaration=True
                )
                print(
                    f"✨ NEW   {nef_name} ★{rating} Pick={pick} "
                    f"Color={label_display or ''} "
                    f"KW={'Y' if (write_keywords and lr_keywords) else 'N'}"
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CSV → Lightroom XMP batch (lr_* contract-driven)"
    )

    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--csv-glob", type=str, default=CSV_GLOB)
    p.add_argument("--image-root", type=Path, default=BASE_DIRECTORY_ROOT)

    # 対象日付（探索rootとCSVをこの日付に固定）
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="YYYY-MM-DD (探索rootとCSVをこの日付に固定)",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        default=DRY_RUN,
        help="writeしない（表示のみ）",
    )
    p.add_argument("--no-backup", action="store_true", help=".xmp.bak を作らない")

    # 強制更新系
    p.add_argument("--force-rating", action="store_true", help="Rating を強制上書き")
    p.add_argument(
        "--force-pick", action="store_true", help="Pick を強制上書き（注意）"
    )
    p.add_argument(
        "--force-color", action="store_true", help="ColorLabel を強制上書き（注意）"
    )
    p.add_argument(
        "--clear-color-if-pick0",
        action="store_true",
        help="pick=0 のとき色を消す（--force-colorと併用推奨）",
    )

    # Pick運用
    p.add_argument(
        "--pick-mode",
        type=str,
        default=DEFAULT_PICK_MODE,
        choices=["flags", "accepted", "accepted_or_secondary", "none"],
        help="Pickの付与ルール（既定: flags）",
    )

    # Keywords運用
    p.add_argument(
        "--write-keywords",
        action="store_true",
        help="lr_keywords を XMPキーワード(dc:subject)へ追記する",
    )
    p.add_argument(
        "--overwrite-keywords",
        action="store_true",
        help="--write-keywords 時に既存キーワードを上書きする（注意）",
    )

    return p.parse_args()


def main():
    args = parse_args()

    dry_run = bool(args.dry_run)
    backup_xmp = not args.no_backup

    # 既定：運用互換で Rating は常に更新したい場合が多いが、ここは明示化。
    # - 既定OFF（安全）。常時更新したい運用なら CLI で --force-rating を付ける。
    force_rating = bool(args.force_rating)

    # dateがあれば探索rootを固定
    image_root = resolve_image_root(args.image_root, args.date)

    print("=== CSV → Lightroom XMP batch start ===")
    print(f"CSV glob : {args.csv_glob}")
    print(f"Date     : {args.date}")
    print(f"ImageRoot: {image_root}")
    print(f"DryRun   : {dry_run}")
    print(f"Backup   : {backup_xmp}")
    print(
        f"Force    : rating={force_rating} pick={args.force_pick} "
        f"color={args.force_color} clear_if_pick0={args.clear_color_if_pick0}"
    )
    print(f"PickMode : {args.pick_mode}")
    print(f"Keywords : write={args.write_keywords} overwrite={args.overwrite_keywords}")
    print("=====================================")

    csv_path = find_csv(args.output_dir, args.csv_glob, args.date)
    print(f"Using CSV: {csv_path}")

    print("Indexing NEF files...")
    nef_index = build_nef_index(image_root)
    print(f"Indexed NEF files: {len(nef_index)}")
    print("=====================================")

    process_csv(
        csv_path,
        nef_index,
        dry_run=dry_run,
        backup_xmp=backup_xmp,
        force_rating=force_rating,
        force_pick=bool(args.force_pick),
        force_color=bool(args.force_color),
        clear_color_if_pick0=bool(args.clear_color_if_pick0),
        pick_mode=args.pick_mode,
        write_keywords=bool(args.write_keywords),
        overwrite_keywords=bool(args.overwrite_keywords),
    )

    print("=== done ===")


if __name__ == "__main__":
    main()
