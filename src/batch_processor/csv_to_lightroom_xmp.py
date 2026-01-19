import argparse
import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Dict

# =========================================================
# 設定（CLIで上書き可能）
# =========================================================

OUTPUT_DIR = Path("output")
CSV_GLOB = "evaluation_ranking_*.csv"

DRY_RUN = False
BACKUP_XMP = True

BASE_DIRECTORY_ROOT = Path("/mnt/l/picture/")

# Pick 判定（ジャンル別に調整可能）
PICK_THRESHOLD_BY_GENRE = {
    "portrait": 65,
    "landscape": 70,
    "snapshot": 60,
    "default": 65,
}

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
}

ET.register_namespace("x", NS["x"])
ET.register_namespace("rdf", NS["rdf"])
ET.register_namespace("xmp", NS["xmp"])
ET.register_namespace("xmpDM", NS["xmpDM"])
ET.register_namespace("photoshop", NS["photoshop"])
ET.register_namespace("lr", NS["lr"])


# =========================================================
# 抽象スコア → Lightroom マッピング
# =========================================================

def score_to_rating(overall: float) -> int:
    """★ = 再編集価値（ジャンル非依存）"""
    if overall >= 80:
        return 4
    if overall >= 70:
        return 3
    if overall >= 60:
        return 2
    if overall >= 50:
        return 1
    return 0


def map_pick(
    *,
    overall: float,
    category: str,
    accepted_flag: int = 0,
    thresholds: Dict[str, float] = PICK_THRESHOLD_BY_GENRE,
) -> int:
    """
    Pick 判定（ジャンル別閾値）
    1) accepted_flag==1ならPick=1（互換）
    2) overall>=threshold(category)ならPick=1
    3) それ以外Pick=0
    """
    if accepted_flag == 1:
        return 1

    cat = (category or "default").strip().lower()
    th = thresholds.get(cat, thresholds.get("default", 65))
    return 1 if overall >= float(th) else 0


def map_color(
    *,
    overall: float,
    technical: float,
    face: float,
    comp: float,
    pick: int,
    category: str,
) -> str:
    """
    Green  : 最終候補
    Blue   : 要確認
    Yellow : 低優先
    """
    if pick != 1:
        return "Yellow"
    if overall >= 80:
        return "Green"
    return "Blue"


# =========================================================
# CSV最新版: lr_color_label 対応
# =========================================================

COLOR_LABEL_MAP = {
    "green": ("green", "グリーン"),
    "yellow": ("yellow", "イエロー"),
    "blue": ("blue", "ブルー"),
    "red": ("red", "レッド"),
    "purple": ("purple", "パープル"),
}

def normalize_lr_color_label(lr_color_label: str) -> Tuple[Optional[str], Optional[str]]:
    if not lr_color_label:
        return None, None
    key = lr_color_label.strip().lower()
    return COLOR_LABEL_MAP.get(key, (key, None))


# =========================================================
# ユーティリティ
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
            # （必要ならここでraiseにしてもOK）
            continue
        index[name] = p

    if dup:
        # 最低限の警告。必要なら raise にしても良い
        sample = list(dup.items())[:5]
        print("⚠️ Duplicate NEF names detected under the search root. (showing up to 5)")
        for name, paths in sample:
            print(f"  - {name}:")
            for pp in paths:
                print(f"      {pp}")

    return index


def get_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, None)
        if value in (None, ""):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def get_int(row: dict, key: str, default: int = 0) -> int:
    try:
        value = row.get(key, None)
        if value in (None, ""):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def get_str(row: dict, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return str(value)


# =========================================================
# XMP 操作ユーティリティ
# =========================================================

def find_target_description(root: ET.Element) -> Optional[ET.Element]:
    for desc in root.findall(".//rdf:Description", NS):
        # 属性運用が主なので、子要素の有無ではなく「見つかったらそれ」でよいが、
        # 互換のためにそれっぽいもの優先のロジックは残す
        for child in desc:
            if child.tag.startswith(f"{{{NS['xmp']}}}") or \
               child.tag.startswith(f"{{{NS['lr']}}}") or \
               child.tag.startswith(f"{{{NS['xmpDM']}}}") or \
               child.tag.startswith(f"{{{NS['photoshop']}}}"):
                return desc
        # 子要素が無いrdf:Descriptionでも属性がある場合があるので最後に返す候補にする
        return desc

    return root.find(".//rdf:Description", NS)


def create_new_xmp(
    rating: int,
    pick: int,
    label_key: Optional[str],
    label_display: Optional[str],
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

    return xmpmeta


def _clear_color_attrs(desc: ET.Element) -> None:
    k = f"{{{NS['photoshop']}}}LabelColor"
    l = f"{{{NS['xmp']}}}Label"
    if k in desc.attrib:
        del desc.attrib[k]
    if l in desc.attrib:
        del desc.attrib[l]


def merge_into_existing_xmp(
    xmp_path: Path,
    rating: int,
    pick: int,
    label_key: Optional[str],
    label_display: Optional[str],
    *,
    dry_run: bool,
    force_rating: bool,
    force_pick: bool,
    force_color: bool,
    clear_color_if_pick0: bool,
):
    tree = ET.parse(xmp_path)
    root = tree.getroot()

    desc = find_target_description(root)
    if desc is None:
        raise RuntimeError("rdf:Description not found in XMP")

    # 1) ★ Rating：force_ratingがTrueなら上書き（デフォルトON運用）
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
        # 強制更新時：
        # - pick=0 で clear_color_if_pick0 なら消す（label_key/label_displayが無い前提でも消える）
        if pick == 0 and clear_color_if_pick0:
            _clear_color_attrs(desc)
        else:
            # label_key/display が None の場合は「何もしない」(既存保持) に倒す
            if label_key:
                desc.set(f"{{{NS['photoshop']}}}LabelColor", label_key)
            if label_display:
                desc.set(f"{{{NS['xmp']}}}Label", label_display)
    else:
        # 通常運用：既存色があるなら守る。無い時だけ付ける。
        if not has_existing_color:
            if label_key:
                desc.set(f"{{{NS['photoshop']}}}LabelColor", label_key)
            if label_display:
                desc.set(f"{{{NS['xmp']}}}Label", label_display)

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
):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            nef_name = get_str(row, "file_name")
            if not nef_name:
                print("⚠️ file_name missing, skip row")
                continue

            overall = get_float(row, "overall_score", 0.0)
            technical = get_float(row, "score_technical", 0.0)
            face = get_float(row, "score_face", 0.0)
            comp = get_float(row, "score_composition", 0.0)
            accepted_flag = get_int(row, "accepted_flag", 0)
            category = get_str(row, "category", "default")

            nef_path = nef_index.get(nef_name)
            if nef_path is None or not nef_path.exists():
                print(f"❌ NEF not found: {nef_name}")
                continue

            xmp_path = nef_path.with_suffix(".xmp")

            # ★ Rating: CSVが指定しているならそれを採用、なければ計算
            lr_rating = get_int(row, "lr_rating", -1)
            rating = lr_rating if lr_rating >= 0 else score_to_rating(overall)

            # Pick
            pick = map_pick(
                overall=overall,
                category=category,
                accepted_flag=accepted_flag,
            )

            # Color:
            #  ここでは「評価結果CSV側の lr_color_label をそのまま信じる」方針にする。
            #  Pick は「採用フラグ」として別概念（色のフィルタ条件には使わない）。
            lr_color_label = get_str(row, "lr_color_label", "")
            color_key, color_disp = normalize_lr_color_label(lr_color_label)

            # CSV 側が空（""）のときは「自動では色を付けない」= None に倒す。
            # 既存色をどうするかは merge_into_existing_xmp() の
            # force_color / clear_color_if_pick0 のポリシーに任せる。
            if not color_key:
                color_key, color_disp = None, None


            if xmp_path.exists():
                if backup_xmp and not dry_run:
                    shutil.copy(xmp_path, xmp_path.with_suffix(".xmp.bak"))

                merge_into_existing_xmp(
                    xmp_path,
                    rating,
                    pick,
                    color_key,
                    color_disp,
                    dry_run=dry_run,
                    force_rating=force_rating,
                    force_pick=force_pick,
                    force_color=force_color,
                    clear_color_if_pick0=clear_color_if_pick0,
                )

                print(f"🔁 MERGE {nef_name} ★{rating} Pick={pick} Color={color_disp}")
            else:
                if dry_run:
                    print(f"[DRY] NEW {nef_name} ★{rating} Pick={pick} Color={color_disp}")
                    continue

                xmp = create_new_xmp(rating, pick, color_key, color_disp)
                ET.ElementTree(xmp).write(xmp_path, encoding="utf-8", xml_declaration=True)
                print(f"✨ NEW   {nef_name} ★{rating} Pick={pick} Color={color_disp}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV → Lightroom XMP batch")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--csv-glob", type=str, default=CSV_GLOB)
    p.add_argument("--image-root", type=Path, default=BASE_DIRECTORY_ROOT)

    # 対象日付（最重要）
    p.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (探索rootとCSVをこの日付に固定)")

    p.add_argument("--dry-run", action="store_true", default=DRY_RUN, help="writeしない（表示のみ）")
    p.add_argument("--no-backup", action="store_true", help=".xmp.bak を作らない")

    # 強制更新系
    p.add_argument("--force-rating", action="store_true", help="Rating を強制上書き（未指定でも既定ON運用）")
    p.add_argument("--force-pick", action="store_true", help="Pick を強制上書き（注意）")
    p.add_argument("--force-color", action="store_true", help="ColorLabel を強制上書き（注意）")
    p.add_argument("--clear-color-if-pick0", action="store_true", help="pick=0 のとき色を消す（--force-colorと併用推奨）")

    return p.parse_args()


def main():
    args = parse_args()

    # 既定：今まで通り Rating は常に更新（force-rating未指定でもON）
    force_rating = True

    dry_run = bool(args.dry_run)
    backup_xmp = not args.no_backup

    # dateがあれば探索rootを固定
    image_root = resolve_image_root(args.image_root, args.date)

    print("=== CSV → Lightroom XMP batch start ===")
    print(f"CSV glob : {args.csv_glob}")
    print(f"Date     : {args.date}")
    print(f"ImageRoot: {image_root}")
    print(f"DryRun   : {dry_run}")
    print(f"Backup   : {backup_xmp}")
    print(f"Force    : rating={force_rating} pick={args.force_pick} color={args.force_color} clear_if_pick0={args.clear_color_if_pick0}")
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
        force_pick=args.force_pick,
        force_color=args.force_color,
        clear_color_if_pick0=args.clear_color_if_pick0,
    )

    print("=== done ===")


if __name__ == "__main__":
    main()
