# tools/gen_evaluator_md.py
# -*- coding: utf-8 -*-

"""
テンプレート + JSON仕様ファイルから Markdown を自動生成するツール。

もともとは Evaluator 設計書専用だったが、
Contract系（例: MetricGradeContract）も生成できるよう汎用化する。

使い方例:
  # 従来どおり（evaluator）
  python tools/gen_evaluator_md.py --spec docs/specs/local_contrast.json

  # 契約ドキュメント（grade contract）
  python tools/gen_evaluator_md.py \
    --spec docs/specs/metric_grade_contract.json \
    --template docs/templates/metric_grade_contract.md.tpl \
    --out-dir docs/contracts \
    --output-name MetricGradeContract.md
"""

from __future__ import annotations

from pathlib import Path
import json
import argparse

from jinja2 import Template


# ============================
# defaults (backward compatible)
# ============================
DEFAULT_TEMPLATE_PATH = Path("docs/templates/evaluator.md.tpl")
DEFAULT_OUTPUT_DIR = Path("docs/evaluators")

DEFAULT_REQUIRED_KEYS = [
    "slug",
    "name",
    "overview",
    "purpose",
    "algorithm",
    "normalization",
    "fallback",
    "testing",
    "future",
    "philosophy",
    "fields",
    "history",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown generator (template + JSON spec)")

    parser.add_argument(
        "--spec",
        required=True,
        help="JSON形式の仕様ファイルパス（例: docs/specs/local_contrast.json）",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE_PATH),
        help=f"テンプレートファイルパス（default: {DEFAULT_TEMPLATE_PATH}）",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"出力先ディレクトリ（default: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="出力ファイル名を明示指定（例: MetricGradeContract.md）。未指定なら slug.md",
    )
    parser.add_argument(
        "--required-keys",
        default="",
        help=(
            "required keys を JSON で指定（例: '[\"slug\",\"title\",\"sections\"]'）。"
            "未指定なら evaluator 用の既定キーを使う。"
        ),
    )

    args = parser.parse_args()

    spec_path = Path(args.spec)
    template_path = Path(args.template)
    out_dir = Path(args.out_dir)

    if not spec_path.exists():
        raise FileNotFoundError(f"spec file not found: {spec_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"template not found: {template_path}")

    spec = load_json(spec_path)

    # required keys
    if args.required_keys:
        try:
            required_keys = json.loads(args.required_keys)
            if not isinstance(required_keys, list) or not all(isinstance(x, str) for x in required_keys):
                raise ValueError
        except Exception:
            raise ValueError("--required-keys must be a JSON list of strings")
    else:
        required_keys = DEFAULT_REQUIRED_KEYS

    missing = [k for k in required_keys if k not in spec]
    if missing:
        raise ValueError(f"missing keys in spec: {missing}")

    template_text = template_path.read_text(encoding="utf-8")
    template = Template(template_text)

    markdown_text = template.render(**spec)

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_name:
        output_path = out_dir / args.output_name
    else:
        slug = str(spec.get("slug"))
        output_path = out_dir / f"{slug}.md"

    output_path.write_text(markdown_text, encoding="utf-8")
    print(f"[OK] generated: {output_path}")


if __name__ == "__main__":
    main()
