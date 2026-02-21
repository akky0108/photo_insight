#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apply_noise_suggestions.py

score_dist_tune の出力（new_params_used.json など）から、
noise の提案閾値(thresholds_5bin)を運用SSOT（config.noise.good_sigma/warn_sigma）へ反映する。

A方針（sigma軸SSOT）:
- good_sigma = thresholds_5bin[1]
- warn_sigma = thresholds_5bin[3]

安全設計:
- デフォルトは dry-run（差分表示のみ）
- --apply を付けたときだけ上書き
- デフォでバックアップ .bak を作る（--no-backup で無効化）
- raw_spec が lower_is_better でない場合は warn（--strict なら停止）

使い方:
  python tools/apply_noise_suggestions.py \
    --params temp/score_dist_tune_out/new_params_used.json \
    --config config/evaluator_thresholds.yaml

  # 実適用
  python tools/apply_noise_suggestions.py \
    --params temp/score_dist_tune_out/new_params_used.json \
    --config config/evaluator_thresholds.yaml \
    --apply
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass(frozen=True)
class NoiseSigmas:
    good_sigma: float
    warn_sigma: float
    source: str
    raw_direction: Optional[str]
    raw_transform: Optional[str]


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_noise_sigmas_from_params(data: Dict[str, Any]) -> NoiseSigmas:
    """
    new_params_used.json の想定構造:
      {
        "noise": {
          "thresholds": [t0,t1,t2,t3],
          "source": "...",
          "raw_spec": { "raw_direction": "...", "raw_transform": "..." }
        },
        ...
      }

    NOTE:
      score_dist_tune の emit-config-yaml は noise_suggestions を吐くが、
      applyツールは new_params_used.json をSSOT入力として扱う（1ファイルで完結させるため）。
    """
    if "noise" not in data or not isinstance(data["noise"], dict):
        raise ValueError("params json missing 'noise' object")

    noise = data["noise"]
    thr = noise.get("thresholds")
    if not isinstance(thr, list) or len(thr) != 4:
        raise ValueError(
            "params json noise.thresholds must be a list of 4 floats [t0,t1,t2,t3]"
        )

    t0, t1, t2, t3 = [float(x) for x in thr]
    # A: sigma軸（lower_is_better）に対して
    good_sigma = t1
    warn_sigma = t3

    source = str(noise.get("source", ""))
    rs = noise.get("raw_spec") if isinstance(noise.get("raw_spec"), dict) else {}
    raw_direction = rs.get("raw_direction")
    raw_transform = rs.get("raw_transform")

    return NoiseSigmas(
        good_sigma=good_sigma,
        warn_sigma=warn_sigma,
        source=source,
        raw_direction=str(raw_direction) if raw_direction is not None else None,
        raw_transform=str(raw_transform) if raw_transform is not None else None,
    )


def unified_diff(old: str, new: str, fromfile: str, tofile: str) -> str:
    diff = difflib.unified_diff(
        old.splitlines(True),
        new.splitlines(True),
        fromfile=fromfile,
        tofile=tofile,
    )
    return "".join(diff)


# -----------------------------
# YAML handling (preferred)
# -----------------------------
def update_config_dict(cfg: Dict[str, Any], sig: NoiseSigmas) -> Dict[str, Any]:
    out = dict(cfg)  # shallow copy
    noise = out.get("noise")
    if not isinstance(noise, dict):
        noise = {}
        out["noise"] = noise

    # SSOT
    noise["good_sigma"] = float(sig.good_sigma)
    noise["warn_sigma"] = float(sig.warn_sigma)

    # 任意: 追跡のため残す（邪魔なら消してOK）
    # noise.setdefault("_meta", {})
    # if isinstance(noise["_meta"], dict):
    #     noise["_meta"]["applied_from"] = "apply_noise_suggestions.py"
    #     noise["_meta"]["source"] = sig.source

    return out


# -----------------------------
# Text patch fallback (no PyYAML)
# -----------------------------
def _format_float(x: float) -> str:
    # 余計に丸めない（そのまま repr 風に）
    return f"{float(x):.16g}"


def patch_yaml_text_config(text: str, sig: NoiseSigmas) -> str:
    """
    PyYAMLがない場合の最小パッチ:
      - noise: ブロックがあればその中の good_sigma / warn_sigma を更新 or 追加
      - noise: ブロックがなければ末尾に追加

    制約:
      - YAMLのフルパースはしない（ここで壊れるような複雑YAMLはPyYAML推奨）
    """
    lines = text.splitlines(True)

    # find top-level "noise:" line (no indentation)
    noise_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("noise:") and (
            ln.strip() == "noise:" or ln.startswith("noise:\n")
        ):
            noise_idx = i
            break

    good_line = f"  good_sigma: {_format_float(sig.good_sigma)}\n"
    warn_line = f"  warn_sigma: {_format_float(sig.warn_sigma)}\n"

    if noise_idx is None:
        # append new block
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        if lines and lines[-1].strip() != "":
            lines.append("\n")
        lines.extend(["noise:\n", good_line, warn_line])
        return "".join(lines)

    # find noise block range: until next top-level key (no indentation, endswith ':')
    start = noise_idx
    end = len(lines)
    for j in range(noise_idx + 1, len(lines)):
        ln = lines[j]
        if ln and not ln.startswith(" ") and ln.rstrip().endswith(":"):
            end = j
            break

    block = lines[start:end]

    def upsert_key(block_lines: list[str], key: str, new_line: str) -> list[str]:
        key_prefix = f"  {key}:"
        found = False
        out_lines: list[str] = []
        inserted = False

        for ln in block_lines:
            if ln.startswith(key_prefix):
                out_lines.append(new_line)
                found = True
            else:
                out_lines.append(ln)

        if not found:
            # insert after "noise:" line; keep any comments immediately after header
            out2: list[str] = []
            out2.append(out_lines[0])  # "noise:\n"
            k_inserted = False
            for ln in out_lines[1:]:
                if (not k_inserted) and (ln.startswith("  #") or ln.strip() == ""):
                    out2.append(ln)
                else:
                    if not k_inserted:
                        out2.append(new_line)
                        k_inserted = True
                    out2.append(ln)
            if not k_inserted:
                out2.append(new_line)
            return out2

        return out_lines

    block2 = upsert_key(block, "good_sigma", good_line)
    block3 = upsert_key(block2, "warn_sigma", warn_line)

    return "".join(lines[:start] + block3 + lines[end:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--params", required=True, help="new_params_used.json (score_dist_tune output)"
    )
    ap.add_argument("--config", required=True, help="config/evaluator_thresholds.yaml")
    ap.add_argument("--apply", action="store_true", help="actually write changes")
    ap.add_argument("--no-backup", action="store_true", help="disable .bak backup")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="fail if raw_spec is not lower_is_better for noise",
    )
    args = ap.parse_args()

    params_path = Path(args.params)
    config_path = Path(args.config)

    data = load_json(params_path)
    sig = extract_noise_sigmas_from_params(data)

    # sanity
    if sig.raw_direction and sig.raw_direction != "lower_is_better":
        msg = (
            f"[WARN] noise raw_spec.raw_direction is '{sig.raw_direction}', expected 'lower_is_better' (sigma axis). "
            f"source={sig.source}"
        )
        if args.strict:
            eprint(msg.replace("[WARN]", "[ERROR]"))
            return 2
        eprint(msg)

    if sig.raw_transform and sig.raw_transform != "lower_is_better":
        msg = (
            f"[WARN] noise raw_spec.raw_transform is '{sig.raw_transform}', expected 'lower_is_better'. "
            f"source={sig.source}"
        )
        if args.strict:
            eprint(msg.replace("[WARN]", "[ERROR]"))
            return 2
        eprint(msg)

    old_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""

    if yaml is not None and old_text.strip():
        # YAML parse/write
        try:
            cfg = yaml.safe_load(old_text)  # type: ignore
            if cfg is None:
                cfg = {}
            if not isinstance(cfg, dict):
                raise ValueError("config yaml root must be a mapping")
            new_cfg = update_config_dict(cfg, sig)
            new_text = yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False)  # type: ignore
        except Exception as e:
            eprint(f"[WARN] PyYAML parse failed; fallback to text patch. ({e})")
            new_text = patch_yaml_text_config(old_text, sig)
    else:
        # no yaml lib or empty file -> text patch
        new_text = patch_yaml_text_config(old_text, sig)

    diff = unified_diff(
        old_text, new_text, fromfile=str(config_path), tofile=str(config_path)
    )
    if diff.strip():
        print(diff)
    else:
        print("[INFO] no changes (already up to date)")
        return 0

    if not args.apply:
        print("[INFO] dry-run (use --apply to write)")
        return 0

    # write with backup
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and (not args.no_backup):
        bak = config_path.with_suffix(config_path.suffix + ".bak")
        shutil.copy2(config_path, bak)
        print(f"[INFO] backup: {bak}")

    config_path.write_text(new_text, encoding="utf-8")
    print(
        f"[INFO] applied: good_sigma={sig.good_sigma}, warn_sigma={sig.warn_sigma} -> {config_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
