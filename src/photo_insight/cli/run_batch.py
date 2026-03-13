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
        CLI 引数 `--processor` または pipeline stage として指定された文字列。

    Returns
    -------
    Type[BaseBatchProcessor]
        解決された Processor クラス。
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
        以下どちらかの形式のクラス指定文字列。

        - package.module:ClassName
        - package.module.ClassName

    Returns
    -------
    Type[BaseBatchProcessor]
        解決された Processor クラス。

    Raises
    ------
    ValueError
        dotted path の形式が不正な場合。
    TypeError
        指定されたクラスが BaseBatchProcessor を継承していない場合。
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
    現時点で正式サポートしている stage は以下。

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
    `--pipeline` 引数を解析し、stage 名リストへ変換する。

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
        `_parse_pipeline_spec()` により生成された stage 名リスト。

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
    文字列値を可能な範囲で Python 値へ変換する。

    Parameters
    ----------
    v : str
        CLI から受け取った値文字列。

    Returns
    -------
    Any
        変換後の値。

    Notes
    -----
    変換ルールは以下。

    - "true" / "false" -> bool
    - 数値文字列 -> int / float
    - Python literal -> ast.literal_eval
    - それ以外 -> str のまま
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
    """
    kwargs: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            i += 1
            continue

        key = token[2:].replace("-", "_").strip()

        if key in _RESERVED_UNKNOWN_KEYS:
            raise ValueError(f"'{token}' is a reserved runner/CLI option and " "cannot be passed as an unknown arg.")

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
        `_parse_unknown_args()` により生成された kwargs。

    Returns
    -------
    Dict[str, Any]
        processor インスタンスに注入する属性辞書。

    Notes
    -----
    dry-run 時でも副作用が起きないよう、instance 生成前に抽出する。

    抽出対象例

    - --date
    - --run-date
    - --target-dir
    - nef_*
    """
    injected: Dict[str, Any] = {}

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

    target_dir = None
    if "target_dir" in exec_kwargs:
        target_dir = exec_kwargs.pop("target_dir")
    elif "dir" in exec_kwargs:
        target_dir = exec_kwargs.pop("dir")

    if target_dir is not None:
        injected["target_dir"] = str(target_dir)

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
        実行対象の processor インスタンス。

    injected : Dict[str, Any]
        `_extract_runtime_overrides()` が返す override 値。

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
        setattr(proc, "target_date", injected["date"])

    if "target_dir" in injected:
        setattr(proc, "target_dir", Path(injected["target_dir"]))

    for k, v in injected.items():
        if k.startswith("nef_"):
            setattr(proc, k, v)


# -------------------------
# Pipeline artifact helpers
# -------------------------
def _resolve_session_name_from_injected(injected: Dict[str, Any]) -> str:
    """
    runtime override 情報から session 名を解決する。

    Parameters
    ----------
    injected : Dict[str, Any]
        `_extract_runtime_overrides()` が返す辞書。

    Returns
    -------
    str
        session 名。
    """
    target_dir = injected.get("target_dir")
    if target_dir:
        return Path(str(target_dir)).name

    date = injected.get("date")
    if date:
        return str(date)

    return "ALL"


def _infer_nef_output_csv_path(
    proc: BaseBatchProcessor,
    injected: Dict[str, Any],
) -> Optional[str]:
    """
    NEF stage 実行後の output CSV パスを推定する。

    Parameters
    ----------
    proc : BaseBatchProcessor
        実行済みの NEF processor インスタンス。

    injected : Dict[str, Any]
        runtime override 情報。

    Returns
    -------
    Optional[str]
        解決できた CSV パス。見つからない場合は None。
    """
    session = _resolve_session_name_from_injected(injected)
    fname = f"{session}_raw_exif_data.csv"

    run_ctx = getattr(proc, "run_ctx", None)
    if run_ctx is not None and getattr(run_ctx, "out_dir", None):
        p1 = Path(run_ctx.out_dir) / "artifacts" / "nef" / session / fname
        if p1.exists():
            return str(p1)

    project_root = Path(getattr(proc, "project_root", Path.cwd()))
    p2 = project_root / "runs" / "latest" / "nef" / session / fname
    if p2.exists():
        return str(p2)

    return None


def _infer_processed_count(proc: BaseBatchProcessor) -> Optional[int]:
    """
    processor インスタンスから処理件数らしき値を推定する。

    Parameters
    ----------
    proc : BaseBatchProcessor
        実行済み processor インスタンス。

    Returns
    -------
    Optional[int]
        推定できた処理件数。取得できない場合は None。
    """
    candidate_names = (
        "processed_count",
        "processed_images",
        "images_processed",
        "total_processed",
        "count_processed",
    )

    for name in candidate_names:
        value = getattr(proc, name, None)
        if isinstance(value, int):
            return value

    run_ctx = getattr(proc, "run_ctx", None)
    if run_ctx is not None:
        for name in candidate_names:
            value = getattr(run_ctx, name, None)
            if isinstance(value, int):
                return value

    return None


def _build_stage_result(
    processor_spec: str,
    proc: BaseBatchProcessor,
    injected: Dict[str, Any],
    exec_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    stage 実行結果の最小メタ情報を構築する。

    Parameters
    ----------
    processor_spec : str
        実行した processor 指定文字列。

    proc : BaseBatchProcessor
        実行済み processor インスタンス。

    injected : Dict[str, Any]
        runtime override 情報。

    exec_kwargs : Optional[Dict[str, Any]]
        実際に processor.execute() へ渡した kwargs。

    Returns
    -------
    Dict[str, Any]
        stage result。
    """
    stage_name = _normalize_pipeline_stage_name(processor_spec)

    result: Dict[str, Any] = {
        "name": stage_name,
        "status": "success",
        "input_csv_path": None,
        "output_csv_path": None,
        "processed_count": None,
        "applied_max_images": None,
        "message": None,
    }

    if exec_kwargs is not None:
        result["input_csv_path"] = exec_kwargs.get("input_csv_path")
        result["applied_max_images"] = exec_kwargs.get("max_images")

    result["processed_count"] = _infer_processed_count(proc)

    if stage_name == "nef":
        result["output_csv_path"] = _infer_nef_output_csv_path(proc, injected)
        if result["output_csv_path"] is None:
            result["message"] = "NEF output CSV path could not be resolved"

    return result


def _build_pipeline_summary(
    stages: List[str],
    stage_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    pipeline 実行結果から summary を構築する。

    Parameters
    ----------
    stages : List[str]
        実行対象 pipeline stage の正規化済みリスト。

    stage_results : List[Dict[str, Any]]
        各 stage の実行結果。

    Returns
    -------
    Dict[str, Any]
        pipeline summary。
    """
    return {
        "pipeline": stages,
        "status": "success",
        "stages": stage_results,
    }


def _print_pipeline_summary(summary: Dict[str, Any]) -> None:
    """
    pipeline summary を標準出力へ表示する。

    Parameters
    ----------
    summary : Dict[str, Any]
        `_build_pipeline_summary()` が返す summary。
    """
    print("[pipeline summary]")
    print(f"pipeline = {','.join(summary['pipeline'])}")
    print(f"status = {summary['status']}")

    for stage in summary["stages"]:
        print(f"[stage] {stage['name']}")
        print(f"  status = {stage['status']}")

        if stage.get("applied_max_images") is not None:
            print(f"  applied_max_images = {stage['applied_max_images']}")

        if stage.get("input_csv_path"):
            print(f"  input_csv_path = {stage['input_csv_path']}")

        if stage.get("output_csv_path"):
            print(f"  output_csv_path = {stage['output_csv_path']}")

        if stage.get("processed_count") is not None:
            print(f"  processed_count = {stage['processed_count']}")

        if stage.get("message"):
            print(f"  message = {stage['message']}")


# -------------------------
# Processor / Pipeline execution
# -------------------------
def run_single_processor(
    processor_spec: str,
    ctor_kwargs: Dict[str, Any],
    exec_kwargs: Dict[str, Any],
    injected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    単一 processor を実行する。

    Parameters
    ----------
    processor_spec : str
        実行対象 processor の指定文字列。
        例: "nef", "portrait_quality", "pkg.mod:ClassName"

    ctor_kwargs : Dict[str, Any]
        processor コンストラクタに渡す引数。

    exec_kwargs : Dict[str, Any]
        processor.execute() に渡す引数。

    injected : Dict[str, Any]
        processor インスタンスへ属性注入する runtime override。

    Returns
    -------
    Dict[str, Any]
        stage result。

    Notes
    -----
    例外はここでは握りつぶさず、そのまま呼び出し側へ送出する。
    """
    Processor = resolve_processor(processor_spec)
    proc = Processor(**ctor_kwargs)
    _apply_runtime_overrides(proc, injected)
    proc.execute(**exec_kwargs)
    return _build_stage_result(processor_spec, proc, injected, exec_kwargs=exec_kwargs)


def run_pipeline_chain(
    stages: List[str],
    ctor_kwargs: Dict[str, Any],
    exec_kwargs: Dict[str, Any],
    injected: Dict[str, Any],
) -> Dict[str, Any]:
    """
    pipeline chain を順次実行する。

    Parameters
    ----------
    stages : List[str]
        実行する stage 名の順序付きリスト。
        例: ["nef", "portrait_quality"]

    ctor_kwargs : Dict[str, Any]
        各 processor のコンストラクタに渡す共通引数。

    exec_kwargs : Dict[str, Any]
        各 processor.execute() に渡す共通引数。

    injected : Dict[str, Any]
        各 processor に注入する共通 runtime override。

    Returns
    -------
    Dict[str, Any]
        pipeline summary。

    Notes
    -----
    PR4 では以下を扱う。

    - stage result の収集
    - pipeline summary の生成
    - max_images の pipeline 対応
      - 先頭 stage のみに適用
      - 後続 stage は upstream artifact に従う
    """
    previous_result: Optional[Dict[str, Any]] = None
    stage_results: List[Dict[str, Any]] = []

    for idx, stage_name in enumerate(stages):
        stage_exec_kwargs = dict(exec_kwargs)

        # max_images は pipeline 先頭 stage にのみ適用し、
        # 後続 stage は upstream artifact の件数に従わせる。
        if idx > 0:
            stage_exec_kwargs.pop("max_images", None)

        if stage_name == "portrait_quality":
            if previous_result is None or previous_result.get("name") != "nef":
                raise RuntimeError("portrait_quality stage requires a previous nef stage result")

            nef_csv = previous_result.get("output_csv_path")
            if not nef_csv:
                raise FileNotFoundError("NEF output CSV path could not be resolved after nef stage execution")

            stage_exec_kwargs["input_csv_path"] = nef_csv

        previous_result = run_single_processor(
            processor_spec=stage_name,
            ctor_kwargs=dict(ctor_kwargs),
            exec_kwargs=stage_exec_kwargs,
            injected=dict(injected),
        )
        stage_results.append(previous_result)

    return _build_pipeline_summary(stages=stages, stage_results=stage_results)


# -------------------------
# CLI
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    CLI 引数パーサを生成する。

    Returns
    -------
    argparse.ArgumentParser
        `run_batch` 用の ArgumentParser。

    対応引数
    --------
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

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve processor/pipeline and kwargs then exit",
    )
    return p


def resolve_processor(spec: str) -> Type[BaseBatchProcessor]:
    """
    processor 指定文字列から Processor クラスを解決する。

    Parameters
    ----------
    spec : str
        エイリアスまたは dotted path。

    Returns
    -------
    Type[BaseBatchProcessor]
        解決された Processor クラス。
    """
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
    --processor と --pipeline は排他。

    - 両方指定 -> エラー
    - 両方未指定 -> エラー
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
        None の場合は sys.argv[1:] を使用する。

    Returns
    -------
    int
        終了コード。

    処理フロー
    ----------
    1. CLI 引数解析
    2. processor / pipeline モード判定
    3. runtime override 抽出
    4. 単段実行または pipeline 実行
    """
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)

    _validate_entrypoint_args(args)

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
    injected = _extract_runtime_overrides(exec_kwargs)

    if args.pipeline:
        stages = _parse_pipeline_spec(args.pipeline)
        _validate_supported_pipeline(stages)

        if args.dry_run:
            print(f"[dry-run] pipeline = {stages}")
            print(f"[dry-run] ctor_kwargs = {ctor_kwargs}")
            print(f"[dry-run] exec_kwargs = {exec_kwargs}")
            print(f"[dry-run] injected = {injected}")
            return 0

        summary = run_pipeline_chain(
            stages=stages,
            ctor_kwargs=ctor_kwargs,
            exec_kwargs=exec_kwargs,
            injected=injected,
        )
        _print_pipeline_summary(summary)
        return 0

    assert args.processor is not None

    Processor = resolve_processor(args.processor)

    if args.dry_run:
        print(f"[dry-run] processor = {Processor.__module__}.{Processor.__name__}")
        print(f"[dry-run] ctor_kwargs = {ctor_kwargs}")
        print(f"[dry-run] exec_kwargs = {exec_kwargs}")
        print(f"[dry-run] injected = {injected}")
        return 0

    run_single_processor(
        processor_spec=args.processor,
        ctor_kwargs=ctor_kwargs,
        exec_kwargs=exec_kwargs,
        injected=injected,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
