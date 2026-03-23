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
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from github import Auth, Github, GithubException

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
    id: str
    title: str
    body: str
    labels: list[str]
    children: list[str]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def default_issues_yml_path() -> Path:
    return repo_root_from_script() / "docs" / "operations" / "github" / "issues.yml"


# -------------------------
# YAML
# -------------------------
def normalize_labels(labels: list[Any]) -> list[str]:
    out: list[str] = []
    for x in labels:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())

    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def load_issues_yml(path: Path) -> list[IssueSpec]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    items = data.get("issues")
    section_name = "issues"

    if items is None:
        items = data.get("epics")
        section_name = "epics"

    if items is None:
        items = []

    if not isinstance(items, list):
        raise ValueError("issues.yml: 'issues' or 'epics' must be a list")

    specs: list[IssueSpec] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"issues.yml: {section_name}[{i}] must be a dict")

        raw_id = it.get("id")
        if not raw_id or not isinstance(raw_id, str):
            raise ValueError(f"issues.yml: {section_name}[{i}] missing required 'id' (external_id)")

        title = it.get("title")
        if not title or not isinstance(title, str):
            raise ValueError(f"issues.yml: {section_name}[{i}] missing valid 'title'")

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
                id=raw_id.strip(),
                title=title.strip(),
                body=body,
                labels=normalize_labels(labels),
                children=[c.strip() for c in children if isinstance(c, str) and c.strip()],
            )
        )

    return specs


def build_spec_lookup(specs: list[IssueSpec]) -> dict[str, IssueSpec]:
    mapping: dict[str, IssueSpec] = {}
    for spec in specs:
        if spec.id in mapping:
            raise ValueError(f"duplicate id in issues.yml: {spec.id}")
        mapping[spec.id] = spec
    return mapping


# -------------------------
# body managed/human handling
# -------------------------
def extract_external_id(body: str) -> Optional[str]:
    m = EXTERNAL_ID_RE.search(body or "")
    return m.group(1) if m else None


def build_managed_section(
    *,
    external_id: str,
    spec_body: str,
    subtasks_md: str = "",
    parent_ref_md: str = "",
) -> str:
    lines: list[str] = [f"<!-- photo_insight:external_id={external_id} -->", ""]

    if spec_body.strip():
        lines.append(spec_body.rstrip())
        lines.append("")

    if parent_ref_md.strip():
        lines.extend(["---", "", parent_ref_md.strip(), ""])

    if subtasks_md.strip():
        lines.extend(["---", "", subtasks_md.strip(), ""])

    return "\n".join(lines).rstrip() + "\n"


def split_sections(body: str) -> tuple[Optional[str], Optional[str], str]:
    text = body or ""
    managed = None
    human = None

    def _extract(block_start: str, block_end: str, s: str) -> tuple[Optional[str], str]:
        if block_start not in s or block_end not in s:
            return None, s

        start_idx = s.find(block_start)
        end_idx = s.find(block_end, start_idx + len(block_start))
        if end_idx == -1:
            return None, s

        inner = s[start_idx + len(block_start) : end_idx]
        new_s = s[:start_idx] + s[end_idx + len(block_end) :]
        return inner.strip("\n"), new_s

    managed, text2 = _extract(MANAGED_START, MANAGED_END, text)
    human, text3 = _extract(HUMAN_START, HUMAN_END, text2)
    rest = text3.strip("\n")
    return managed, human, rest


def compose_full_body(*, new_managed: str, existing_body: Optional[str]) -> str:
    old = existing_body or ""
    _old_managed, old_human, old_rest = split_sections(old)

    if (old_human is None or old_human.strip() == "") and old_rest.strip():
        preserved_human = old_rest.strip("\n")
    else:
        preserved_human = (old_human or "").strip("\n")

    parts: list[str] = [
        MANAGED_START,
        new_managed.rstrip(),
        MANAGED_END,
        "",
        HUMAN_START,
    ]

    if preserved_human:
        parts.append(preserved_human.rstrip())
    else:
        parts.append("")

    parts.append(HUMAN_END)
    return "\n".join(parts).rstrip() + "\n"


# -------------------------
# GitHub helpers
# -------------------------
def ensure_labels(
    repo: Any,
    existing_labels: dict[str, Any],
    label_names: list[str],
    *,
    dry_run: bool,
) -> list[Any]:
    label_objects: list[Any] = []

    for name in label_names:
        if name not in existing_labels:
            info(f"label missing -> create: {name}")
            if dry_run:
                existing_labels[name] = None
            else:
                try:
                    new_label = repo.create_label(name=name, color=DEFAULT_LABEL_COLOR)
                    existing_labels[name] = new_label
                except GithubException as exc:
                    warn(f"label create failed: {name} ({exc})")
                    existing_labels[name] = None

        obj = existing_labels.get(name)
        if obj is not None:
            label_objects.append(obj)

    return label_objects


def fetch_existing_by_external_id(
    repo: Any,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    ext_to_issue: dict[str, Any] = {}
    dups: dict[str, list[int]] = {}

    for issue in repo.get_issues(state="all"):
        ext = extract_external_id(issue.body or "")
        if not ext:
            continue

        if ext in ext_to_issue:
            dups.setdefault(ext, [ext_to_issue[ext].number])
            dups[ext].append(issue.number)
            continue

        ext_to_issue[ext] = issue

    return ext_to_issue, dups


# -------------------------
# Rendering subtasks / parent ref
# -------------------------
def render_subtasks(parent: IssueSpec, child_issues: list[Any]) -> str:
    if not parent.children:
        return ""

    lines: list[str] = ["## Subtasks", ""]
    for child_issue in child_issues:
        lines.append(f"- [ ] #{child_issue.number} {child_issue.title}")
    return "\n".join(lines).strip() + "\n"


def render_parent_ref(parent_issue: Any) -> str:
    return f"Parent: #{parent_issue.number} {parent_issue.title}"


def resolve_token() -> str:
    token = (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or "").strip()
    if token:
        return token
    raise ValueError("環境変数 GITHUB_TOKEN または GH_TOKEN が設定されていません")


# -------------------------
# Sync core
# -------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="akky0108/photo_insight", help="owner/repo")
    ap.add_argument(
        "--issues-yml",
        default=str(default_issues_yml_path()),
        help="path to issues.yml",
    )
    ap.add_argument(
        "--env-file",
        default="",
        help="optional .env file path",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="do not create/update anything, just print actions",
    )
    args = ap.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=True)
    else:
        load_dotenv(override=True)

    token = resolve_token()

    issues_file = Path(args.issues_yml).resolve()
    if not issues_file.exists():
        raise FileNotFoundError(f"issues.yml not found: {issues_file}")

    github_client = Github(auth=Auth.Token(token))
    repo = github_client.get_repo(args.repo)

    specs = load_issues_yml(issues_file)
    spec_by_id = build_spec_lookup(specs)

    existing_labels = {label.name: label for label in repo.get_labels()}
    ext_to_issue, dups = fetch_existing_by_external_id(repo)

    if dups:
        err("Duplicate external_id detected in existing GitHub Issues. Please resolve first:")
        for ext, nums in dups.items():
            err(f"  external_id={ext}: issues={nums}")
        return 2

    failed: list[str] = []

    def upsert(
        spec: IssueSpec,
        *,
        subtasks_md: str = "",
        parent_ref_md: str = "",
    ) -> Optional[Any]:
        managed = build_managed_section(
            external_id=spec.id,
            spec_body=spec.body,
            subtasks_md=subtasks_md,
            parent_ref_md=parent_ref_md,
        )

        existing = ext_to_issue.get(spec.id)
        label_names = spec.labels

        if args.dry_run:
            if existing:
                info(
                    f"DRY-RUN update: #{existing.number} {spec.title} " f"(external_id={spec.id}) labels={label_names}"
                )
            else:
                info(f"DRY-RUN create: {spec.title} " f"(external_id={spec.id}) labels={label_names}")
            return existing

        try:
            ensure_labels(repo, existing_labels, label_names, dry_run=False)

            if existing:
                new_body = compose_full_body(
                    new_managed=managed,
                    existing_body=existing.body or "",
                )
                existing.edit(title=spec.title, body=new_body, labels=label_names)
                info(f"UPDATED: #{existing.number} {spec.title}")
                return existing

            new_body = compose_full_body(new_managed=managed, existing_body="")
            issue = repo.create_issue(
                title=spec.title,
                body=new_body,
                labels=label_names,
            )
            info(f"CREATED: #{issue.number} {spec.title}")
            ext_to_issue[spec.id] = issue
            return issue

        except GithubException as exc:
            err(f"upsert failed: {spec.title} (external_id={spec.id}) ({exc})")
            return None

    for spec in specs:
        issue = upsert(spec)
        if issue is None and not args.dry_run:
            failed.append(spec.id)

    if args.dry_run:
        info("DONE. Dry-run mode: no issues/labels were created/updated.")
        return 0

    parents = [spec for spec in specs if spec.children]
    for parent in parents:
        parent_issue = ext_to_issue.get(parent.id)
        if not parent_issue:
            warn(f"parent issue missing after pass1: {parent.id} {parent.title}")
            continue

        child_issues: list[Any] = []
        for child_id in parent.children:
            child_spec = spec_by_id.get(child_id)
            if child_spec is None:
                warn(f"parent '{parent.title}' references unknown child id: {child_id}")
                continue

            child_issue = ext_to_issue.get(child_id)
            if not child_issue:
                warn(f"child issue missing after pass1: {child_id} ({child_spec.title})")
                continue

            child_issues.append(child_issue)
            upsert(child_spec, parent_ref_md=render_parent_ref(parent_issue))

        upsert(parent, subtasks_md=render_subtasks(parent, child_issues))

    if failed:
        warn(f"Some issues failed: {failed}")

    info("DONE.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
