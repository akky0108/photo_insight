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
    "xmpDM": "http://ns.adobe.com/xmp/1.0/DynamicMedia/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",  # 探索用に残してOK
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
    thresholds: dict[str, float] = PICK_THRESHOLD_BY_GENRE,
) -> int:
    """
    Pick 判定（ジャンル別閾値）

    優先順位:
      1) accepted_flag == 1 なら強制 Pick=1（既存ロジック互換）
      2) それ以外は overall >= threshold(category) なら Pick=1
      3) それ以外は Pick=0

    ※Reject(-1)は出さない（Lightroom事故防止）
    """
    # 既存の「合格」フラグがあるなら最優先
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
    Lightroom ColorLabel mapping

    Green  : 最終候補（人がほぼ確実に残す）
    Blue   : 要確認
    Yellow : 低優先
    """
    # Pickでないなら色は基本付けない想定だけど、
    # process_csv側で pick!=1 は color=None にしてるならここはPick前提でOK
    if pick != 1:
        return "Yellow"

    # 例：強い候補だけGreen、それ以外はBlue（ここは好みで調整）
    if overall >= 80:
        return "Green"
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


def set_attr(desc: ET.Element, ns_key: str, name: str, value: Optional[str]) -> None:
    """rdf:Description に XMPプロパティを属性としてセット（Lightroom互換を優先）"""
    q = f"{{{NS[ns_key]}}}{name}"
    if value is None or str(value).strip() == "":
        # ラベル無しは「属性を消す」方が事故りにくい
        if q in desc.attrib:
            del desc.attrib[q]
        return
    desc.set(q, str(value))


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


def create_new_xmp(rating: int, pick: int, label_key: Optional[str], label_display: Optional[str]) -> ET.Element:
    xmpmeta = ET.Element(f"{{{NS['x']}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, f"{{{NS['rdf']}}}RDF")

    desc = ET.SubElement(
        rdf,
        f"{{{NS['rdf']}}}Description",
        attrib={f"{{{NS['rdf']}}}about": ""},
    )

    # ★ Rating（属性）
    desc.set(f"{{{NS['xmp']}}}Rating", str(int(rating)))

    # Pick（属性: xmpDM:pick）
    desc.set(f"{{{NS['xmpDM']}}}pick", str(int(pick)))

    # Color label（属性）
    if label_key:
        desc.set(f"{{{NS['photoshop']}}}LabelColor", label_key)
    if label_display:
        desc.set(f"{{{NS['xmp']}}}Label", label_display)

    return xmpmeta


def merge_into_existing_xmp(
    xmp_path: Path,
    rating: int,
    pick: int,
    label_key: Optional[str],
    label_display: Optional[str],
):
    tree = ET.parse(xmp_path)
    root = tree.getroot()

    desc = find_target_description(root)
    if desc is None:
        raise RuntimeError("rdf:Description not found in XMP")

    # 1) ★ Rating：常に上書き（属性）
    desc.set(f"{{{NS['xmp']}}}Rating", str(int(rating)))

    # 2) Pick：未設定(0) or 無し のときだけ上書き（属性 xmpDM:pick）
    existing_pick = (desc.get(f"{{{NS['xmpDM']}}}pick") or "").strip()
    if existing_pick in ("", "0"):
        desc.set(f"{{{NS['xmpDM']}}}pick", str(int(pick)))

    # 3) Color：既存があれば守る（属性）
    existing_label = (desc.get(f"{{{NS['xmp']}}}Label") or "").strip()
    existing_key = (desc.get(f"{{{NS['photoshop']}}}LabelColor") or "").strip()
    if not (existing_label or existing_key):
        if label_key:
            desc.set(f"{{{NS['photoshop']}}}LabelColor", label_key)
        if label_display:
            desc.set(f"{{{NS['xmp']}}}Label", label_display)

    # お掃除：lr:Pick / lr:ColorLabel があれば消す（混乱源）
    for tag in (f"{{{NS['lr']}}}Pick", f"{{{NS['lr']}}}ColorLabel"):
        node = desc.find(tag)
        if node is not None:
            desc.remove(node)

    if not DRY_RUN:
        tree.write(xmp_path, encoding="utf-8", xml_declaration=True)


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

            # ---------------------------------
            # Lightroom マッピング（単一責務）
            # ---------------------------------
            # ★ Rating: CSVが指定しているならそれを採用、なければ計算
            lr_rating = get_int(row, "lr_rating", -1)
            rating = lr_rating if lr_rating >= 0 else score_to_rating(overall)

            # Pick: 閾値×ジャンル（accepted_flagは上書き用途で優先）
            pick = map_pick(
                overall=overall,
                category=category,
                accepted_flag=accepted_flag,
            )

            # Color: Pick=1のときだけ“提案”する。CSVに指定があればそれを優先。
            label_key = get_str(row, "lr_labelcolor_key", "").strip().lower()      # green/yellow/...
            label_display = get_str(row, "lr_label_display", "").strip()          # グリーン/イエロー/...

            # 既存XMPなら merge_into_existing_xmp(..., label_key, label_display)
            # 新規なら create_new_xmp(..., label_key, label_display)

            if pick == 1:
                color_key = label_key if label_key else None
                color_display = label_display if label_display else None
            else:
                color_key = None
                color_display = None

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
                    color_key,
                    color_display,
                )

                print(
                    f"🔁 MERGE {nef_name} "
                    f"★{rating} Pick={pick} Color={color_display}"
                )
            else:
                if DRY_RUN:
                    print(
                        f"[DRY] NEW {nef_name} "
                        f"★{rating} Pick={pick} Color={color_display}"
                    )
                    continue

                xmp = create_new_xmp(rating, pick, color_key, color_display)
                ET.ElementTree(xmp).write(
                    xmp_path,
                    encoding="utf-8",
                    xml_declaration=True,
                )

                print(
                    f"✨ NEW   {nef_name} "
                    f"★{rating} Pick={pick} Color={color_display}"
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
