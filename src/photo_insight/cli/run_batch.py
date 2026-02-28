# src/photo_insight/cli/run_batch.py
from __future__ import annotations

import argparse
import importlib
import sys
import ast
import re
from pathlib import Path
from typing import Any, Dict, Type, Optional

from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor

# -------------------------
# Reserved keys (runner-owned)
# -------------------------
# These keys are owned by the runner/CLI and must NOT be passed via unknown args.
# NOTE: run_date/date are handled separately as runtime overrides
#       (allowed in unknown args).
_RESERVED_UNKNOWN_KEYS = {
    "processor",
    "config",
    "config_path",
    "max_workers",
    "config_env",
    "config_paths",
    "dry_run",
}


# -------------------------
# Processor registry (lazy)
# -------------------------
def _load_processor_by_alias(name: str) -> Type[BaseBatchProcessor]:
    """
    Short-name registry. Keep imports lazy to avoid heavy deps at CLI import time.
    """
    key = (name or "").strip().lower()

    if key in ("nef", "nef_file", "nef_file_batch"):
        from photo_insight.pipelines.nef.nef_batch_process import NEFFileBatchProcess

        return NEFFileBatchProcess

    if key in ("evaluation_rank", "rank", "eval_rank"):
        from photo_insight.pipelines.evaluation_rank import EvaluationRankBatchProcessor

        return EvaluationRankBatchProcessor

    if key in ("portrait_quality", "quality", "portrait"):
        from photo_insight.pipelines.portrait_quality import PortraitQualityBatchProcessor

        return PortraitQualityBatchProcessor

    raise KeyError(f"Unknown processor alias: {name}")


def _load_processor_by_dotted_path(dotted: str) -> Type[BaseBatchProcessor]:
    """
    Dotted path format: package.module:ClassName  (preferred)
                      or package.module.ClassName (fallback)
    """
    s = (dotted or "").strip()
    if ":" in s:
        mod, cls = s.split(":", 1)
    else:
        parts = s.split(".")
        if len(parts) < 2:
            raise ValueError(f"Invalid dotted path: {dotted}")
        mod, cls = ".".join(parts[:-1]), parts[-1]

    m = importlib.import_module(mod)
    obj = getattr(m, cls)
    if not isinstance(obj, type):
        raise TypeError(f"{dotted} does not resolve to a class")
    if not issubclass(obj, BaseBatchProcessor):
        raise TypeError(f"{dotted} is not a BaseBatchProcessor subclass")
    return obj


# -------------------------
# Unknown args -> kwargs
# -------------------------
def _coerce_value(v: str) -> Any:
    """
    Best-effort conversion:
    - "true"/"false" -> bool
    - numbers -> int/float
    - python literals (list/dict/tuple/None) via literal_eval
    - else keep string
    """
    if v is None:
        return True

    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"

    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _parse_unknown_args(unknown: list[str]) -> Dict[str, Any]:
    """
    Accepts patterns like:
      --target-dir /path
      --append-mode true
      --exts "['.NEF','.DNG']"
      --flag   (flag -> True)

    Returns kwargs dict with snake_case keys (dashes -> underscores).

    Safety:
    - Runner/CLI-owned keys must not be passed via unknown args.
    """
    kwargs: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            i += 1
            continue

        key = token[2:].replace("-", "_").strip()

        # Disallow collisions with runner/CLI-owned keys
        if key in _RESERVED_UNKNOWN_KEYS:
            raise ValueError(f"'{token}' is a reserved runner/CLI option and " f"cannot be passed as an unknown arg.")

        # flag only
        if i + 1 >= len(unknown) or unknown[i + 1].startswith("--"):
            kwargs[key] = True
            i += 1
            continue

        val = unknown[i + 1]
        kwargs[key] = _coerce_value(val)
        i += 2

    return kwargs


# -------------------------
# Runtime overrides (inject plan / inject to proc)
# -------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _extract_runtime_overrides(exec_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    dry-run でも副作用ゼロで扱えるように、注入対象だけを exec_kwargs から抜き出す。

    例:
      --run-date 2026-01-31  -> injected["date"] = "2026-01-31"
      --date 2026-01-31      -> 同上
    """
    injected: Dict[str, Any] = {}

    # --- date/run_date --
    run_date = None
    if "run_date" in exec_kwargs:
        run_date = exec_kwargs.pop("run_date")
    elif "date" in exec_kwargs:
        run_date = exec_kwargs.pop("date")

    if run_date is not None:
        s = str(run_date).strip()
        if not _DATE_RE.match(s):
            raise ValueError(f"Invalid date format: {s} (expected YYYY-MM-DD)")
        injected["date"] = s

    # --- target_dir ---
    target_dir = None
    if "target_dir" in exec_kwargs:
        target_dir = exec_kwargs.pop("target_dir")
    elif "dir" in exec_kwargs:
        # 互換：昔の --dir を許すなら
        target_dir = exec_kwargs.pop("dir")

    if target_dir is not None:
        injected["target_dir"] = str(target_dir)

    return injected


def _apply_runtime_overrides(proc: BaseBatchProcessor, injected: Dict[str, Any]) -> None:
    """
    本実行時のみ、processor インスタンスに注入する。
    """
    if "date" in injected:
        setattr(proc, "date", injected["date"])

    if "target_dir" in injected:
        # strでもPathでも来るので Path 化
        setattr(proc, "target_dir", Path(injected["target_dir"]))


# -------------------------
# CLI
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_batch",
        description="Unified runner for photo_insight batch processors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--processor",
        required=True,
        help=(
            "Processor alias (nef/evaluation_rank/portrait_quality) OR dotted path (pkg.mod:Class)."  # noqa: E501
        ),
    )

    # common knobs (BaseBatchProcessor ctor)
    p.add_argument(
        "--config",
        dest="config_path",
        default="config/config.prod.yaml",
        help="Config file path",
    )
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--config-env", default=None, help="ConfigManager env name (optional)")
    p.add_argument(
        "--config-paths",
        default=None,
        help="Comma-separated config paths (optional). e.g. config.base.yaml,config.prod.yaml",  # noqa: E501
    )

    p.add_argument("--dry-run", action="store_true", help="Resolve processor and kwargs then exit")
    return p


def resolve_processor(spec: str) -> Type[BaseBatchProcessor]:
    s = spec.strip()
    if ":" in s or s.count(".") >= 2:
        return _load_processor_by_dotted_path(s)
    return _load_processor_by_alias(s)


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    # build ctor kwargs
    config_paths = None
    if args.config_paths:
        config_paths = [x.strip() for x in args.config_paths.split(",") if x.strip()]

    Processor = resolve_processor(args.processor)

    ctor_kwargs: Dict[str, Any] = {
        "config_path": args.config_path,
        "max_workers": args.max_workers,
    }
    if args.config_env is not None:
        ctor_kwargs["config_env"] = args.config_env
    if config_paths is not None:
        ctor_kwargs["config_paths"] = config_paths

    exec_kwargs = _parse_unknown_args(unknown)

    # ★dry-run でも副作用ゼロで注入計画だけ作る（インスタンス化しない）
    injected = _extract_runtime_overrides(exec_kwargs)

    if args.dry_run:
        print(f"[dry-run] processor = {Processor.__module__}.{Processor.__name__}")
        print(f"[dry-run] ctor_kwargs = {ctor_kwargs}")
        print(f"[dry-run] exec_kwargs = {exec_kwargs}")
        print(f"[dry-run] injected = {injected}")
        return 0

    proc = Processor(**ctor_kwargs)
    _apply_runtime_overrides(proc, injected)

    proc.execute(**exec_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
