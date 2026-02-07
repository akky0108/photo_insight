#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
create_issues.py

issues.yml を読み込み、GitHub Issues を作成する。

改善点（完全版）:
- 既存Issue（open/closed両方）との重複をタイトルで検知してスキップ
- 親子(issue.children) を解決して、子を親本文にチェックリストとしてリンク追記
- 既存ラベルの自動作成（色はデフォルト）
- Dry-run 対応
- 失敗しても全体は止めない（例外を握って次へ）

想定する issues.yml 形式:
issues:
  - title: "[#701] ..."
    body: |
      ...
    labels: [enhancement, ...]
    children:
      - child_id_1
      - child_id_2

  - id: child_id_1
    title: "[#701-1] ..."
    body: |
      ...
    labels: [...]

注意:
- 重複判定は「タイトル完全一致」
  既に同じタイトルが存在すれば作成しない（open/closed問わず）
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from github import Github
from github.GithubException import GithubException


DEFAULT_LABEL_COLOR = "f29513"


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    eprint(f"[WARN] {msg}")


def err(msg: str) -> None:
    eprint(f"[ERROR] {msg}")


@dataclass
class IssueSpec:
    # issues.yml の1要素を正規化したもの
    id: Optional[str]
    title: str
    body: str
    labels: List[str]
    children: List[str]


def load_issues_yml(path: str) -> List[IssueSpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    items = data.get("issues", [])
    if not isinstance(items, list):
        raise ValueError("issues.yml: 'issues' must be a list")

    specs: List[IssueSpec] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"issues.yml: issues[{i}] must be a dict")
        title = it.get("title")
        if not title or not isinstance(title, str):
            raise ValueError(f"issues.yml: issues[{i}] missing valid 'title'")
        specs.append(
            IssueSpec(
                id=it.get("id") if isinstance(it.get("id"), str) else None,
                title=title,
                body=it.get("body", "") if isinstance(it.get("body", ""), str) else "",
                labels=list(it.get("labels", [])) if isinstance(it.get("labels", []), list) else [],
                children=list(it.get("children", [])) if isinstance(it.get("children", []), list) else [],
            )
        )
    return specs


def build_child_lookup(specs: List[IssueSpec]) -> Dict[str, IssueSpec]:
    m: Dict[str, IssueSpec] = {}
    for s in specs:
        if s.id:
            if s.id in m:
                warn(f"duplicate child id in yaml: {s.id} (later one wins)")
            m[s.id] = s
    return m


def normalize_labels(labels: List[Any]) -> List[str]:
    out: List[str] = []
    for x in labels:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    # 重複排除（順序維持）
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def get_existing_issue_titles(repo) -> Tuple[set, Dict[str, int]]:
    """
    既存Issueのタイトル集合を作る（open/closed両方）。
    ついでに title -> issue_number のマップも返す（同タイトル複数は先勝ち）。
    """
    titles = set()
    title_to_num: Dict[str, int] = {}

    # open
    for it in repo.get_issues(state="open"):
        t = (it.title or "").strip()
        if not t:
            continue
        titles.add(t)
        title_to_num.setdefault(t, it.number)

    # closed
    for it in repo.get_issues(state="closed"):
        t = (it.title or "").strip()
        if not t:
            continue
        titles.add(t)
        title_to_num.setdefault(t, it.number)

    return titles, title_to_num


def ensure_labels(repo, existing_labels: Dict[str, Any], label_names: List[str], *, dry_run: bool) -> List[Any]:
    label_objects = []
    for name in label_names:
        if name not in existing_labels:
            info(f"label missing -> create: {name}")
            if dry_run:
                # ダミーはNoneでよい（create_issueに渡さない運用にする）
                existing_labels[name] = None
            else:
                try:
                    new_label = repo.create_label(name=name, color=DEFAULT_LABEL_COLOR)
                    existing_labels[name] = new_label
                except GithubException as e:
                    warn(f"label create failed: {name} ({e})")
                    # 作れなくてもIssue作成は続ける（labels無しで）
                    existing_labels[name] = None

        obj = existing_labels.get(name)
        if obj is not None:
            label_objects.append(obj)
        else:
            # ラベル作成失敗 or dry-run のためスキップ
            pass
    return label_objects


def render_parent_body_with_children(parent: IssueSpec, child_specs: List[IssueSpec]) -> str:
    """
    親Issue本文末尾に、子Issueチェックリスト（リンクは作成後に追記したいが、
    作成前はリンクが無いので「タイトルのみ」で出す）。
    """
    if not child_specs:
        return parent.body

    lines: List[str] = []
    lines.append(parent.body.rstrip())
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Subtasks")
    lines.append("")
    for c in child_specs:
        lines.append(f"- [ ] {c.title}")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="akky0108/photo_insight", help="owner/repo")
    ap.add_argument("--issues-yml", default="", help="path to issues.yml (default: script_dir/issues.yml)")
    ap.add_argument("--dry-run", action="store_true", help="do not create anything, just print actions")
    ap.add_argument("--skip-duplicates", action="store_true", default=True, help="skip issues if title already exists (default on)")
    ap.add_argument("--only-open-duplicates", action="store_true",
                    help="if set, only check duplicates against open issues (default checks open+closed)")
    args = ap.parse_args()

    # env
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("環境変数 GITHUB_TOKEN が設定されていません")

    # path resolve
    script_dir = os.path.dirname(os.path.abspath(__file__))
    issues_file = args.issues_yml or os.path.join(script_dir, "issues.yml")
    if not os.path.exists(issues_file):
        raise FileNotFoundError(f"issues.yml not found: {issues_file}")

    # github
    g = Github(token)
    repo = g.get_repo(args.repo)

    specs = load_issues_yml(issues_file)
    child_lookup = build_child_lookup(specs)

    # labels cache
    existing_labels = {label.name: label for label in repo.get_labels()}

    # duplicates cache
    if args.only_open_duplicates:
        existing_titles = set()
        title_to_num: Dict[str, int] = {}
        for it in repo.get_issues(state="open"):
            t = (it.title or "").strip()
            if not t:
                continue
            existing_titles.add(t)
            title_to_num.setdefault(t, it.number)
    else:
        existing_titles, title_to_num = get_existing_issue_titles(repo)

    # 作成順: 親（childrenを持つ or id無し）→ 子（idあり）
    # ただし、YAML上の順序も尊重したいので「親っぽいものを先に作る」程度に留める
    def is_child_only(s: IssueSpec) -> bool:
        return bool(s.id) and not s.children

    parents = [s for s in specs if not is_child_only(s)]
    children_only = [s for s in specs if is_child_only(s)]

    created_title_to_issue = {}  # title -> issue (PyGithub object)
    created_children_by_parent_title: Dict[str, List[Any]] = {}

    def create_one(spec: IssueSpec, *, body_override: Optional[str] = None) -> Optional[Any]:
        title = spec.title.strip()
        if args.skip_duplicates and title in existing_titles:
            num = title_to_num.get(title)
            info(f"SKIP duplicate: {title}" + (f" (#{num})" if num else ""))
            return None

        label_names = normalize_labels(spec.labels)
        if args.dry_run:
            info(f"DRY-RUN create issue: {title}")
            if label_names:
                info(f"  labels: {label_names}")
            if body_override is not None:
                info("  body: (overridden)")
            return None

        try:
            label_objects = ensure_labels(repo, existing_labels, label_names, dry_run=False)
            issue = repo.create_issue(
                title=title,
                body=body_override if body_override is not None else spec.body,
                labels=label_objects,
            )
            info(f"CREATED: #{issue.number} {title}")
            existing_titles.add(title)
            title_to_num.setdefault(title, issue.number)
            return issue
        except GithubException as e:
            err(f"create failed: {title} ({e})")
            return None

    # 1) 親作成（子チェックリストはタイトルだけで埋め込み）
    for p in parents:
        child_specs: List[IssueSpec] = []
        for cid in p.children:
            cs = child_lookup.get(cid)
            if cs is None:
                warn(f"parent '{p.title}' references unknown child id: {cid}")
                continue
            child_specs.append(cs)

        body = render_parent_body_with_children(p, child_specs)
        issue = create_one(p, body_override=body)
        if issue is not None:
            created_title_to_issue[p.title] = issue
            created_children_by_parent_title[p.title] = []

    # 2) 子作成（親が作成されていれば、子本文に親リンクを入れる）
    #    ※「親の本文に子リンクを後から追記」も可能だが、PyGithubでIssue編集が必要になるので
    #      ここでは「子の本文に親参照を入れる」だけにする（運用上十分）
    parent_title_to_issue = created_title_to_issue

    # 親が既存で作られている可能性もあるので、親リンク付与は title_to_num から引ければ行う
    for c in children_only:
        # 親探索: この子を children に含む親specを探す
        parent_candidates = [p for p in parents if (c.id and c.id in p.children)]
        parent_ref = ""
        if parent_candidates:
            # 先頭を採用（通常は1つ）
            pt = parent_candidates[0].title
            pnum = title_to_num.get(pt)
            if pnum:
                parent_ref = f"\n\n---\n\nParent: #{pnum} {pt}\n"
            else:
                parent_ref = f"\n\n---\n\nParent: {pt}\n"

        body = (c.body.rstrip() + parent_ref).strip() + "\n"
        issue = create_one(c, body_override=body)
        if issue is not None and parent_candidates:
            created_children_by_parent_title[parent_candidates[0].title].append(issue)

    info("DONE.")
    if args.dry_run:
        info("Dry-run mode: no issues/labels were created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
