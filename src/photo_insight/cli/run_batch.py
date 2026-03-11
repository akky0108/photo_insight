# src/photo_insight/cli/run_batch.py
from __future__ import annotations

import argparse
import ast
import importlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from photo_insight.core.batch_framework.base_batch import BaseBatchProcessor

# -------------------------
# Reserved keys (runner-owned)
# -------------------------
# These keys are owned by the runner/CLI and must NOT be passed via unknown args.
# NOTE: run_date/date are handled separately as runtime overrides
#       (allowed in unknown args).
_RESERVED_UNKNOWN_KEYS = {
    "processor",
    "pipeline",
    "config",
    "config_path",
    "max_workers",
    "config_env",
    "config_paths",
    "dry_run",
}

# PR1: currently supported pipeline definitions
_SUPPORTED_PIPELINES: set[tuple[str, ...]] = {
    ("nef", "portrait_quality"),
}


# -------------------------
# Processor registry (lazy)
# -------------------------
def _load_processor_by_alias(name: str) -> Type[BaseBatchProcessor]:
    """
    processor のエイリアス名から Processor クラスを解決する。

    Parameters
    ----------
    name : str
        CLI 引数 `--processor` で指定された文字列。

    Returns
    -------
    Type[BaseBatchProcessor]
        解決された Processor クラス。

    Notes
    -----
    import は CLI 起動時の依存を軽くするため **lazy import** で行う。
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
    dotted path から Processor クラスを解決する。

    Parameters
    ----------
    dotted : str
        形式は以下のどちらか。

        package.module:ClassName
        package.module.ClassName

    Returns
    -------
    Type[BaseBatchProcessor]
        解決された Processor クラス。

    Raises
    ------
    ValueError
        dotted path の形式が不正な場合
    TypeError
        指定されたクラスが BaseBatchProcessor を継承していない場合
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
# Pipeline helpers
# -------------------------
def _normalize_pipeline_stage_name(name: str) -> str:
    """
    pipeline stage 名を正規化する。

    Parameters
    ----------
    name : str
        `--pipeline` 引数で指定された stage 名。

    Returns
    -------
    str
        正規化された stage 名。

    Notes
    -----
    PR1 時点では以下のみ正式対応。

    - "nef"
    - "portrait_quality"
    """
    key = (name or "").strip().lower()

    if key in ("nef", "nef_file", "nef_file_batch"):
        return "nef"

    if key in ("portrait_quality", "quality", "portrait"):
        return "portrait_quality"

    if key in ("evaluation_rank", "rank", "eval_rank"):
        return "evaluation_rank"

    raise ValueError(f"Unknown pipeline stage: {name}")


def _parse_pipeline_spec(spec: str) -> List[str]:
    """
    `--pipeline` 引数を解析し stage のリストへ変換する。

    Parameters
    ----------
    spec : str
        CLI 引数 `--pipeline` の値。

        例:
        "nef,portrait_quality"

    Returns
    -------
    List[str]
        正規化された stage 名リスト。

        例:
        ["nef", "portrait_quality"]
    """
    if spec is None:
        raise ValueError("--pipeline must not be None")

    items = [x.strip() for x in spec.split(",") if x.strip()]
    if not items:
        raise ValueError("--pipeline must not be empty")

    stages = [_normalize_pipeline_stage_name(x) for x in items]
    return stages


def _validate_supported_pipeline(stages: List[str]) -> None:
    """
    pipeline がサポート対象かを検証する。

    Parameters
    ----------
    stages : List[str]
        `_parse_pipeline_spec` により生成された stage 名リスト。

    Raises
    ------
    ValueError
        サポートされていない pipeline の場合。
    """
    key = tuple(stages)
    if key not in _SUPPORTED_PIPELINES:
        supported = ", ".join(",".join(x) for x in sorted(_SUPPORTED_PIPELINES))
        raise ValueError(f"Unsupported pipeline: {','.join(stages)}. " f"Currently supported pipeline(s): {supported}")


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
    CLI の未知引数（unknown args）を processor 用 kwargs に変換する。

    Parameters
    ----------
    unknown : list[str]
        argparse が解釈できなかった CLI 引数。

    Returns
    -------
    Dict[str, Any]
        processor.execute() に渡される kwargs。

    Examples
    --------

    --target-dir /path
    → {"target_dir": "/path"}

    --append-mode true
    → {"append_mode": True}

    --flag
    → {"flag": True}
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
            raise ValueError(f"'{token}' is a reserved runner/CLI option and cannot be passed as an unknown arg.")

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
    runtime override 引数を抽出する。

    Parameters
    ----------
    exec_kwargs : Dict[str, Any]
        `_parse_unknown_args` により生成された kwargs。

    Returns
    -------
    Dict[str, Any]
        processor インスタンスに注入する属性。

    Notes
    -----
    dry-run 時でも副作用が起きないよう
    instance 生成前に抽出する。

    対象例

    --date
    --run-date
    --target-dir
    nef_*
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

    # --- NEF specific (IMPORTANT): do NOT pass to BaseBatchProcessor.process ---
    for k in list(exec_kwargs.keys()):
        if k.startswith("nef_"):
            injected[k] = exec_kwargs.pop(k)

    return injected


def _apply_runtime_overrides(proc: BaseBatchProcessor, injected: Dict[str, Any]) -> None:
    """
    runtime override を processor インスタンスへ注入する。

    Parameters
    ----------
    proc : BaseBatchProcessor
        実行する processor インスタンス。

    injected : Dict[str, Any]
        `_extract_runtime_overrides` が返す override 値。

    Notes
    -----
    以下の属性が processor に設定される。

    - date
    - target_date
    - target_dir
    - nef_*
    """
    if "date" in injected:
        setattr(proc, "date", injected["date"])
        # ★NEF側が target_date を見る設計に合わせる（最小互換）
        setattr(proc, "target_date", injected["date"])

    if "target_dir" in injected:
        setattr(proc, "target_dir", Path(injected["target_dir"]))

    # ★NEF options: attributes injection
    for k, v in injected.items():
        if k.startswith("nef_"):
            setattr(proc, k, v)


# -------------------------
# CLI
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    CLI 引数パーサを生成する。

    Returns
    -------
    argparse.ArgumentParser

    対応引数

    --processor
        単段 processor 実行

    --pipeline
        pipeline chain 実行

    --config
    --max-workers
    --config-env
    --config-paths
    --dry-run
    """
    p = argparse.ArgumentParser(
        prog="run_batch",
        description="Unified runner for photo_insight batch processors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--processor",
        required=False,
        help=("Processor alias (nef/evaluation_rank/portrait_quality) " "OR dotted path (pkg.mod:Class)."),
    )

    p.add_argument(
        "--pipeline",
        required=False,
        help=("Comma-separated pipeline stages. " "Currently supported: nef,portrait_quality"),
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
        help=("Comma-separated config paths (optional). " "e.g. config.base.yaml,config.prod.yaml"),
    )

    p.add_argument("--dry-run", action="store_true", help="Resolve processor/pipeline and kwargs then exit")
    return p


def resolve_processor(spec: str) -> Type[BaseBatchProcessor]:
    s = spec.strip()
    if ":" in s or s.count(".") >= 2:
        return _load_processor_by_dotted_path(s)
    return _load_processor_by_alias(s)


def _validate_entrypoint_args(args: argparse.Namespace) -> None:
    """
    CLI 引数の整合性を検証する。

    Parameters
    ----------
    args : argparse.Namespace
        argparse が解析した CLI 引数。

    Rules
    -----

    --processor と --pipeline は排他

    両方指定 → エラー
    両方未指定 → エラー
    """
    has_processor = bool(args.processor and args.processor.strip())
    has_pipeline = bool(args.pipeline and args.pipeline.strip())

    if has_processor and has_pipeline:
        raise ValueError("--processor and --pipeline are mutually exclusive")

    if not has_processor and not has_pipeline:
        raise ValueError("Either --processor or --pipeline must be specified")


def main(argv: Optional[list[str]] = None) -> int:
    """
    run_batch CLI のエントリポイント。

    Parameters
    ----------
    argv : Optional[list[str]]
        CLI 引数配列。
        None の場合は sys.argv を使用。

    Returns
    -------
    int
        終了コード

    処理フロー
    ----------

    1. CLI 引数解析
    2. processor / pipeline モード判定
    3. runtime override 抽出
    4. processor 実行

    PR1 段階では pipeline 実行は未実装。
    """
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    _validate_entrypoint_args(args)

    # build ctor kwargs
    config_paths = None
    if args.config_paths:
        config_paths = [x.strip() for x in args.config_paths.split(",") if x.strip()]

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

    # -------------------------
    # Pipeline mode (PR1: validate only, no orchestration yet)
    # -------------------------
    if args.pipeline:
        stages = _parse_pipeline_spec(args.pipeline)
        _validate_supported_pipeline(stages)

        if args.dry_run:
            print(f"[dry-run] pipeline = {stages}")
            print(f"[dry-run] ctor_kwargs = {ctor_kwargs}")
            print(f"[dry-run] exec_kwargs = {exec_kwargs}")
            print(f"[dry-run] injected = {injected}")
            return 0

        raise NotImplementedError(
            "Pipeline execution is not implemented in PR1 yet. "
            "Use --dry-run to validate arguments, or use --processor for single-stage execution. "
            "Pipeline orchestration will be added in the next step."
        )

    # -------------------------
    # Single processor mode
    # -------------------------
    assert args.processor is not None  # for type checkers
    Processor = resolve_processor(args.processor)

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
