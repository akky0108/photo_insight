# tools/gen_evaluator_md.py
# -*- coding: utf-8 -*-

"""
Evaluator 用の設計ドキュメント（Markdown）を
テンプレート + JSON仕様ファイルから自動生成するツール。

目的：
- コピペによる設計書の乱立・ズレを防ぐ
- Evaluator 間でドキュメント構造を統一する
- 将来の修正・拡張を容易にする

使い方：
    python tools/gen_evaluator_md.py --spec docs/specs/local_contrast.json

出力先：
    docs/evaluators/<slug>.md
"""

from pathlib import Path
import json
import argparse

from jinja2 import Template


# ============================
# 設定値
# ============================

# テンプレートファイルのパス
TEMPLATE_PATH = Path("docs/templates/evaluator.md.tpl")

# 出力先ディレクトリ
OUTPUT_DIR = Path("docs/evaluators")


# ============================
# メイン処理
# ============================

def main() -> None:
    """
    エントリーポイント。

    - コマンドライン引数を読む
    - JSON仕様をロード
    - テンプレートに流し込む
    - Markdownを生成して保存
    """

    # ----------------------------
    # 引数定義
    # ----------------------------
    parser = argparse.ArgumentParser(
        description="Evaluator用Markdown設計書ジェネレータ"
    )

    parser.add_argument(
        "--spec",
        required=True,
        help="JSON形式の仕様ファイルパス（例: docs/specs/local_contrast.json）",
    )

    args = parser.parse_args()

    spec_path = Path(args.spec)

    # ----------------------------
    # 入力チェック
    # ----------------------------
    if not spec_path.exists():
        raise FileNotFoundError(f"spec file not found: {spec_path}")

    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"template not found: {TEMPLATE_PATH}")

    # ----------------------------
    # テンプレート読み込み
    # ----------------------------
    with TEMPLATE_PATH.open("r", encoding="utf-8") as f:
        template_text = f.read()

    template = Template(template_text)

    # ----------------------------
    # JSON仕様ファイル読み込み
    # ----------------------------
    with spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    # ----------------------------
    # 必須キーの簡易チェック
    # ----------------------------
    required_keys = [
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

    missing = [k for k in required_keys if k not in spec]
    if missing:
        raise ValueError(f"missing keys in spec: {missing}")

    # ----------------------------
    # テンプレートレンダリング
    # ----------------------------
    # spec の中身をそのままテンプレートに渡す
    markdown_text = template.render(**spec)

    # ----------------------------
    # 出力ディレクトリ作成
    # ----------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 出力ファイル名決定
    # ----------------------------
    # slug をベースにファイル名を作る
    output_path = OUTPUT_DIR / f"{spec['slug']}.md"

    # ----------------------------
    # ファイル書き込み
    # ----------------------------
    output_path.write_text(markdown_text, encoding="utf-8")

    # ----------------------------
    # 完了メッセージ
    # ----------------------------
    print(f"[OK] generated: {output_path}")


# ============================
# 実行エントリ
# ============================

if __name__ == "__main__":
    main()
