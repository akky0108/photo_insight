import csv
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

# =========================================================
# 設定
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
    "lr": "http://ns.adobe.com/lightroom/1.0/",
}

ET.register_namespace("x", NS["x"])
ET.register_namespace("rdf", NS["rdf"])
ET.register_namespace("xmp", NS["xmp"])
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


def map_pick(accepted_flag: int) -> int:
    # accepted_flag == 1 → AI 合格
    return 1 if accepted_flag == 1 else 0


def map_color(
    *,
    overall: float,
    technical: float,
    intent: float,
    pick: int,
    category: str,
) -> str:
    """
    Lightroom ColorLabel mapping

    Green  : 最終候補（人がほぼ確実に残す）
    Blue   : 要確認
    Yellow : 低優先
    """

    # ❌ Pick でないものは Green にしない
    if pick != 1:
        return "Yellow"

    # ✅ Pick の中でさらに強いものだけ Green
    if overall >= 75:
        return "Green"

    # 👀 Pick だが確定ではない
    return "Blue"



# =========================================================
# ユーティリティ
# =========================================================

def build_nef_index(base_dir: Path) -> dict[str, Path]:
    """
    /mnt/l/picture/YYYY/YYYY-MM-DD/*.NEF を全収集
    """
    index = {}
    for p in base_dir.rglob("*.NEF"):
        index[p.name] = p
    return index


def find_latest_csv(output_dir: Path, pattern: str) -> Path:
    csv_files = sorted(
        output_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV found: {output_dir}/{pattern}")
    return csv_files[0]


def get_float(row: dict, key: str, default: float = 0.0) -> float:
    """
    ・キーが存在しない → default
    ・空文字 / None → default
    ・数値変換失敗 → default
    """
    try:
        value = row.get(key, None)
        if value in (None, ""):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def get_int(row: dict, key: str, default: int = 0) -> int:
    """
    int 用（accepted_flag など）
    """
    try:
        value = row.get(key, None)
        if value in (None, ""):
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def get_str(row: dict, key: str, default: str = "") -> str:
    """
    KeyError / None / 空文字対策
    """
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return str(value)


# =========================================================
# XMP 操作ユーティリティ
# =========================================================

def get_or_create(parent: ET.Element, tag: str) -> ET.Element:
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    return node


def find_target_description(root: ET.Element) -> Optional[ET.Element]:
    for desc in root.findall(".//rdf:Description", NS):
        for child in desc:
            if child.tag.startswith(f"{{{NS['xmp']}}}") or \
               child.tag.startswith(f"{{{NS['lr']}}}"):
                return desc
    return root.find(".//rdf:Description", NS)


def create_new_xmp(rating: int, pick: int, color: Optional[str]) -> ET.Element:
    xmpmeta = ET.Element(f"{{{NS['x']}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, f"{{{NS['rdf']}}}RDF")
    desc = ET.SubElement(
        rdf,
        f"{{{NS['rdf']}}}Description",
        attrib={f"{{{NS['rdf']}}}about": ""},
    )

    ET.SubElement(desc, f"{{{NS['xmp']}}}Rating").text = str(rating)
    ET.SubElement(desc, f"{{{NS['lr']}}}Pick").text = str(pick)

    if color:
        ET.SubElement(desc, f"{{{NS['lr']}}}ColorLabel").text = color

    return xmpmeta


def merge_into_existing_xmp(
    xmp_path: Path,
    rating: int,
    pick: int,
    color: Optional[str],
):
    tree = ET.parse(xmp_path)
    root = tree.getroot()

    desc = find_target_description(root)
    if desc is None:
        raise RuntimeError("rdf:Description not found in XMP")

    get_or_create(desc, f"{{{NS['xmp']}}}Rating").text = str(rating)
    get_or_create(desc, f"{{{NS['lr']}}}Pick").text = str(pick)

    if color:
        get_or_create(desc, f"{{{NS['lr']}}}ColorLabel").text = color

    if not DRY_RUN:
        tree.write(
            xmp_path,
            encoding="utf-8",
            xml_declaration=True,
        )

# =========================================================
# メイン処理
# =========================================================

def process_csv(csv_path: Path, nef_index: dict[str, Path]):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # ---------------------------------
            # 安全な値取得（KeyError 完全回避）
            # ---------------------------------
            nef_name = get_str(row, "file_name")
            if not nef_name:
                print("⚠️ file_name missing, skip row")
                continue

            overall = get_float(row, "overall_score", 0.0)
            technical = get_float(row, "technical_score", 0.0)
            intent = get_float(row, "intent_score", 0.0)
            accepted_flag = get_int(row, "accepted_flag", 0)
            category = get_str(row, "category", "default")

            nef_path = nef_index.get(nef_name)
            if nef_path is None or not nef_path.exists():
                print(f"❌ NEF not found: {nef_name}")
                continue

            xmp_path = nef_path.with_suffix(".xmp")

            # ---------------------------------
            # Lightroom マッピング（単一責務）
            # ---------------------------------
            rating = score_to_rating(overall)
            pick = map_pick(accepted_flag)
            color = map_color(
                overall=overall,
                technical=technical,
                intent=intent,
                pick=pick,
                category=category,
            )

            # ---------------------------------
            # XMP 書き込み
            # ---------------------------------
            if xmp_path.exists():
                if BACKUP_XMP and not DRY_RUN:
                    shutil.copy(xmp_path, xmp_path.with_suffix(".xmp.bak"))

                merge_into_existing_xmp(
                    xmp_path,
                    rating,
                    pick,
                    color,
                )

                print(
                    f"🔁 MERGE {nef_name} "
                    f"★{rating} Pick={pick} Color={color}"
                )
            else:
                if DRY_RUN:
                    print(
                        f"[DRY] NEW {nef_name} "
                        f"★{rating} Pick={pick} Color={color}"
                    )
                    continue

                xmp = create_new_xmp(rating, pick, color)
                ET.ElementTree(xmp).write(
                    xmp_path,
                    encoding="utf-8",
                    xml_declaration=True,
                )

                print(
                    f"✨ NEW   {nef_name} "
                    f"★{rating} Pick={pick} Color={color}"
                )


def main():
    print("=== CSV → Lightroom XMP batch start ===")
    print(f"CSV glob : {CSV_GLOB}")
    print(f"ImageRoot: {BASE_DIRECTORY_ROOT}")
    print(f"DryRun   : {DRY_RUN}")
    print("=====================================")

    csv_path = find_latest_csv(OUTPUT_DIR, CSV_GLOB)
    print(f"Using CSV: {csv_path}")

    print("Indexing NEF files...")
    nef_index = build_nef_index(BASE_DIRECTORY_ROOT)
    print(f"Indexed NEF files: {len(nef_index)}")
    print("=====================================")

    process_csv(csv_path, nef_index)

    print("=== done ===")


if __name__ == "__main__":
    main()
