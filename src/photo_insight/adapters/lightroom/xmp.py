from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


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


def find_target_description(root: ET.Element) -> Optional[ET.Element]:
    """
    操作対象の rdf:Description を返す。
    まず Lightroom / XMP 系の子要素を持つ Description を優先し、
    見つからなければ最初の Description を返す。
    """
    for desc in root.findall(".//rdf:Description", NS):
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
    keywords: Optional[list[str]] = None,
) -> ET.Element:
    """
    新規 XMP を生成する。
    """
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
    key_attr = f"{{{NS['photoshop']}}}LabelColor"
    label_attr = f"{{{NS['xmp']}}}Label"

    if key_attr in desc.attrib:
        del desc.attrib[key_attr]
    if label_attr in desc.attrib:
        del desc.attrib[label_attr]


def _get_or_create_bag(desc: ET.Element) -> ET.Element:
    """
    dc:subject/rdf:Bag を取得し、無ければ作る。
    """
    subject = desc.find(f"{{{NS['dc']}}}subject")
    if subject is None:
        subject = ET.SubElement(desc, f"{{{NS['dc']}}}subject")

    bag = subject.find(f"{{{NS['rdf']}}}Bag")
    if bag is None:
        bag = ET.SubElement(subject, f"{{{NS['rdf']}}}Bag")

    return bag


def _existing_keywords(desc: ET.Element) -> list[str]:
    subject = desc.find(f"{{{NS['dc']}}}subject")
    if subject is None:
        return []

    bag = subject.find(f"{{{NS['rdf']}}}Bag")
    if bag is None:
        return []

    out: list[str] = []
    for li in bag.findall(f"{{{NS['rdf']}}}li"):
        if li.text:
            out.append(li.text.strip())
    return out


def _ensure_dc_subject(desc: ET.Element, keywords: list[str], *, overwrite: bool) -> None:
    """
    keywords を dc:subject に反映する。
    overwrite=False の場合は既存キーワードを尊重しつつ、無いものだけ追加する。
    """
    normalized_keywords = [k.strip() for k in keywords if k and str(k).strip()]
    if not normalized_keywords:
        return

    if overwrite:
        subject = desc.find(f"{{{NS['dc']}}}subject")
        if subject is not None:
            desc.remove(subject)

    existing = set(_existing_keywords(desc)) if not overwrite else set()
    bag = _get_or_create_bag(desc)

    for keyword in normalized_keywords:
        if keyword in existing:
            continue
        li = ET.SubElement(bag, f"{{{NS['rdf']}}}li")
        li.text = keyword
        existing.add(keyword)


def merge_into_existing_xmp(
    xmp_path: Path,
    rating: int,
    pick: int,
    label_key: Optional[str],
    label_display: Optional[str],
    *,
    keywords: Optional[list[str]],
    dry_run: bool,
    force_rating: bool,
    force_pick: bool,
    force_color: bool,
    clear_color_if_pick0: bool,
    write_keywords: bool,
    overwrite_keywords: bool,
) -> None:
    """
    既存 XMP に対して rating / pick / color / keywords を反映する。
    """
    tree = ET.parse(xmp_path)
    root = tree.getroot()

    desc = find_target_description(root)
    if desc is None:
        raise RuntimeError("rdf:Description not found in XMP")

    if force_rating:
        desc.set(f"{{{NS['xmp']}}}Rating", str(int(rating)))

    existing_pick = (desc.get(f"{{{NS['xmpDM']}}}pick") or "").strip()
    if force_pick or existing_pick in ("", "0"):
        desc.set(f"{{{NS['xmpDM']}}}pick", str(int(pick)))

    existing_label = (desc.get(f"{{{NS['xmp']}}}Label") or "").strip()
    existing_key = (desc.get(f"{{{NS['photoshop']}}}LabelColor") or "").strip()
    has_existing_color = bool(existing_label or existing_key)

    if force_color:
        if pick == 0 and clear_color_if_pick0:
            _clear_color_attrs(desc)
        else:
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

    if write_keywords and keywords:
        _ensure_dc_subject(desc, keywords, overwrite=overwrite_keywords)

    for tag in (f"{{{NS['lr']}}}Pick", f"{{{NS['lr']}}}ColorLabel"):
        node = desc.find(tag)
        if node is not None:
            desc.remove(node)

    if not dry_run:
        tree.write(xmp_path, encoding="utf-8", xml_declaration=True)
