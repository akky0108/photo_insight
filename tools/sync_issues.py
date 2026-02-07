#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sync_issues.py

issues.yml を SSOT として GitHub Issues を同期（upsert）する。

特徴:
- external_id（issues.yml の id）で既存 Issue を特定（open/closed 両方）
- create / update の冪等同期（何回回しても同じ状態）
- managed セクションだけ上書きし、人間の追記（human セクション）は保持
- 親子関係(children)を解決し、最終的に親本文に子リンク付きチェックリストを反映
- ラベルは自動作成（色はデフォルト）
- dry-run 対応
- external_id が重複している既存 Issue を検知したら安全のため停止

想定 issues.yml:

issues:
  - id: epic_noise_701
    title: "[#701] ..."
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
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from github import Github
from github.GithubException import GithubException

DEFAULT_LABEL_COLOR = "f29513"

# --- body markers ---
EXTERNAL_ID_RE = re.compile(r"<!--\s*photo_insight:external_id=([A-Za-z0-9_\-\.]+)\s*-->")
MANAGED_START = "<!-- managed:start -->"
MANAGED_END = "<!-- managed:end -->"
HUMAN_START = "<!-- human:start -->"
HUMAN_END = "<!-- human:end -->"


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
    id: str  # external_id (required)
    title: str
    body: str
    labels: List[str]
    children: List[str]


# -------------------------
# YAML
# -------------------------
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

        _id = it.get("id")
        if not _id or not isinstance(_id, str):
            raise ValueError(f"issues.yml: issues[{i}] missing required 'id' (external_id)")

        title = it.get("title")
        if not title or not isinstance(title, str):
            raise ValueError(f"issues.yml: issues[{i}] missing valid 'title'")

        body = it.get("body", "")
        if not isinstance(body, str):
            body = ""

        labels = it.get("labels", [])
        if not isinstance(labels, list):
            labels = []

        children = it.get("children", [])
        if not isinstance(children, list):
            children = []

        specs.append(
            IssueSpec(
                id=_id.strip(),
                title=title.strip(),
                body=body,
                labels=normalize_labels(labels),
                children=[c.strip() for c in children if isinstance(c, str) and c.strip()],
            )
        )
    return specs


def normalize_labels(labels: List[Any]) -> List[str]:
    out: List[str] = []
    for x in labels:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    # de-dup preserve order
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def build_spec_lookup(specs: List[IssueSpec]) -> Dict[str, IssueSpec]:
    m: Dict[str, IssueSpec] = {}
    for s in specs:
        if s.id in m:
            raise ValueError(f"duplicate id in issues.yml: {s.id}")
        m[s.id] = s
    return m


# -------------------------
# body managed/human handling
# -------------------------
def extract_external_id(body: str) -> Optional[str]:
    m = EXTERNAL_ID_RE.search(body or "")
    return m.group(1) if m else None


def build_managed_section(*, external_id: str, title: str, spec_body: str, subtasks_md: str = "", parent_ref_md: str = "") -> str:
    """
    managed section is fully generated.
    - embeds external_id marker
    - includes spec body
    - optionally includes parent_ref and subtasks
    """
    lines: List[str] = []
    lines.append(f"<!-- photo_insight:external_id={external_id} -->")
    lines.append("")
    # NOTE: title is not repeated to avoid noise; GitHub title is the title SSOT
    if spec_body.strip():
        lines.append(spec_body.rstrip())
        lines.append("")
    if parent_ref_md.strip():
        lines.append("---")
        lines.append("")
        lines.append(parent_ref_md.strip())
        lines.append("")
    if subtasks_md.strip():
        lines.append("---")
        lines.append("")
        lines.append(subtasks_md.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def split_sections(body: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Return (managed, human, rest).
    - managed: content between managed markers (without markers)
    - human: content between human markers (without markers)
    - rest: any content outside markers that we keep as-is
    """
    text = body or ""
    managed = None
    human = None

    def _extract(block_start: str, block_end: str, s: str) -> Tuple[Optional[str], str]:
        if block_start not in s or block_end not in s:
            return None, s
        # find first pair
        a = s.find(block_start)
        b = s.find(block_end, a + len(block_start))
        if b == -1:
            return None, s
        inner = s[a + len(block_start): b]
        # remove the whole block including markers
        new_s = s[:a] + s[b + len(block_end):]
        return inner.strip("\n"), new_s

    managed, text2 = _extract(MANAGED_START, MANAGED_END, text)
    human, text3 = _extract(HUMAN_START, HUMAN_END, text2)
    rest = text3.strip("\n")
    return managed, human, rest


def compose_full_body(*, new_managed: str, existing_body: Optional[str]) -> str:
    """
    Overwrite managed section; preserve human section if exists; preserve any rest as well.
    If existing has no human section, we keep 'rest' as human notes implicitly by appending it
    into human section (so users don't lose manual edits made before we introduced sections).
    """
    old = existing_body or ""
    _old_managed, old_human, old_rest = split_sections(old)

    # If there was no explicit human section but there is rest text, treat it as human content.
    if (old_human is None or old_human.strip() == "") and old_rest.strip():
        preserved_human = old_rest.strip("\n")
    else:
        preserved_human = (old_human or "").strip("\n")

    parts: List[str] = []
    parts.append(MANAGED_START)
    parts.append(new_managed.rstrip())
    parts.append(MANAGED_END)
    parts.append("")
    parts.append(HUMAN_START)
    if preserved_human:
        parts.append(preserved_human.rstrip())
    else:
        parts.append("")  # keep empty human block
    parts.append(HUMAN_END)
    return "\n".join(parts).rstrip() + "\n"


# -------------------------
# GitHub helpers
# -------------------------
def ensure_labels(repo, existing_labels: Dict[str, Any], label_names: List[str], *, dry_run: bool) -> List[Any]:
    label_objects = []
    for name in label_names:
        if name not in existing_labels:
            info(f"label missing -> create: {name}")
            if dry_run:
                existing_labels[name] = None
            else:
                try:
                    new_label = repo.create_label(name=name, color=DEFAULT_LABEL_COLOR)
                    existing_labels[name] = new_label
                except GithubException as e:
                    warn(f"label create failed: {name} ({e})")
                    existing_labels[name] = None

        obj = existing_labels.get(name)
        if obj is not None:
            label_objects.append(obj)
    return label_objects


def fetch_existing_by_external_id(repo) -> Tuple[Dict[str, Any], Dict[str, List[int]]]:
    """
    Build external_id -> issue_object mapping by scanning all issues.
    Also returns dup map external_id -> [issue_numbers] if duplicates exist.
    """
    ext_to_issue: Dict[str, Any] = {}
    dups: Dict[str, List[int]] = {}

    # PyGithub supports state="all"
    for it in repo.get_issues(state="all"):
        body = it.body or ""
        ext = extract_external_id(body)
        if not ext:
            continue
        if ext in ext_to_issue:
            # duplicate detected
            dups.setdefault(ext, [ext_to_issue[ext].number])
            dups[ext].append(it.number)
            continue
        ext_to_issue[ext] = it

    return ext_to_issue, dups


# -------------------------
# Rendering subtasks / parent ref
# -------------------------
def render_subtasks(parent: IssueSpec, child_issues: List[Any]) -> str:
    if not parent.children:
        return ""

    lines: List[str] = []
    lines.append("## Subtasks")
    lines.append("")
    child_by_id: Dict[str, Any] = {extract_external_id((ci.body or "")) or "": ci for ci in child_issues}
    # But child_issue body might not yet have marker if legacy; safer: use a passed map elsewhere.
    # Here we assume we pass correct child_issues in the same order as parent.children.
    for ci in child_issues:
        lines.append(f"- [ ] #{ci.number} {ci.title}")
    return "\n".join(lines).strip() + "\n"


def render_parent_ref(parent_issue: Any) -> str:
    return f"Parent: #{parent_issue.number} {parent_issue.title}"


# -------------------------
# Sync core
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="akky0108/photo_insight", help="owner/repo")
    ap.add_argument("--issues-yml", default="", help="path to issues.yml (default: script_dir/issues.yml)")
    ap.add_argument("--dry-run", action="store_true", help="do not create/update anything, just print actions")
    args = ap.parse_args()

    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("環境変数 GITHUB_TOKEN が設定されていません")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    issues_file = args.issues_yml or os.path.join(script_dir, "issues.yml")
    if not os.path.exists(issues_file):
        raise FileNotFoundError(f"issues.yml not found: {issues_file}")

    g = Github(token)
    repo = g.get_repo(args.repo)

    specs = load_issues_yml(issues_file)
    spec_by_id = build_spec_lookup(specs)

    # labels cache
    existing_labels = {label.name: label for label in repo.get_labels()}

    # existing issues by external_id
    ext_to_issue, dups = fetch_existing_by_external_id(repo)
    if dups:
        err("Duplicate external_id detected in existing GitHub Issues. Please resolve first:")
        for ext, nums in dups.items():
            err(f"  external_id={ext}: issues={nums}")
        return 2

    # We'll do 2-pass:
    # Pass1: upsert all issues without parent subtasks links (placeholder)
    # Pass2: update parents managed section to include linked subtasks

    created_or_updated: Dict[str, Any] = {}  # external_id -> issue
    failed: List[str] = []

    def upsert(spec: IssueSpec, *, subtasks_md: str = "", parent_ref_md: str = "") -> Optional[Any]:
        # managed content
        managed = build_managed_section(
            external_id=spec.id,
            title=spec.title,
            spec_body=spec.body,
            subtasks_md=subtasks_md,
            parent_ref_md=parent_ref_md,
        )

        existing = ext_to_issue.get(spec.id)
        label_names = spec.labels

        if args.dry_run:
            if existing:
                info(f"DRY-RUN update: #{existing.number} {spec.title} (external_id={spec.id}) labels={label_names}")
            else:
                info(f"DRY-RUN create: {spec.title} (external_id={spec.id}) labels={label_names}")
            return None

        try:
            label_objects = ensure_labels(repo, existing_labels, label_names, dry_run=False)

            if existing:
                new_body = compose_full_body(new_managed=managed, existing_body=existing.body or "")
                # update title/body/labels
                existing.edit(title=spec.title, body=new_body, labels=label_names)
                info(f"UPDATED: #{existing.number} {spec.title}")
                return existing

            # create
            new_body = compose_full_body(new_managed=managed, existing_body="")
            issue = repo.create_issue(title=spec.title, body=new_body, labels=label_names)
            info(f"CREATED: #{issue.number} {spec.title}")
            ext_to_issue[spec.id] = issue
            return issue

        except GithubException as e:
            err(f"upsert failed: {spec.title} (external_id={spec.id}) ({e})")
            return None

    # Pass1: upsert everything (no linked subtasks yet)
    for s in specs:
        issue = upsert(s)
        if issue is None and not args.dry_run:
            failed.append(s.id)
            continue
        if issue is not None:
            created_or_updated[s.id] = issue

    # If dry-run, stop here (we already printed actions)
    if args.dry_run:
        info("DONE. Dry-run mode: no issues/labels were created/updated.")
        return 0

    # Pass2: parent update with linked subtasks + children update with parent ref
    # Build parent list: any spec with children
    parents = [s for s in specs if s.children]
    for p in parents:
        parent_issue = ext_to_issue.get(p.id)
        if not parent_issue:
            warn(f"parent issue missing after pass1: {p.id} {p.title}")
            continue

        child_issues: List[Any] = []
        for cid in p.children:
            cs = spec_by_id.get(cid)
            if cs is None:
                warn(f"parent '{p.title}' references unknown child id: {cid}")
                continue
            child_issue = ext_to_issue.get(cid)
            if not child_issue:
                warn(f"child issue missing after pass1: {cid} ({cs.title})")
                continue
            child_issues.append(child_issue)

            # ensure child has parent ref in managed section
            child_parent_ref = render_parent_ref(parent_issue)
            upsert(cs, parent_ref_md=child_parent_ref)  # re-upsert child with parent ref

        # finally update parent with linked subtasks
        subtasks_md = render_subtasks(p, child_issues)
        upsert(p, subtasks_md=subtasks_md)

    if failed:
        warn(f"Some issues failed: {failed}")

    info("DONE.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
