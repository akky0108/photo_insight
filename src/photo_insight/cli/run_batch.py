from __future__ import annotations

import argparse
import ast
import importlib
import json
import re
import sys
import time
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

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# -------------------------
# Processor registry (lazy)
# -------------------------
def _load_processor_by_alias(name: str) -> Type[BaseBatchProcessor]:
    """
    processor のエイリアス名から Processor クラスを解決する。
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

    形式:
    - package.module:ClassName
    - package.module.ClassName
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


def resolve_processor(spec: str) -> Type[BaseBatchProcessor]:
    """
    processor 指定文字列から Processor クラスを解決する。
    """
    s = spec.strip()
    if ":" in s or s.count(".") >= 2:
        return _load_processor_by_dotted_path(s)
    return _load_processor_by_alias(s)


# -------------------------
# Stage / pipeline helpers
# -------------------------
def _normalize_pipeline_stage_name(name: str) -> str:
    """
    pipeline stage 名を正規化する。
    """
    key = (name or "").strip().lower()

    if key in ("nef", "nef_file", "nef_file_batch"):
        return "nef"

    if key in ("portrait_quality", "quality", "portrait"):
        return "portrait_quality"

    if key in ("evaluation_rank", "rank", "eval_rank"):
        return "evaluation_rank"

    raise ValueError(f"Unknown pipeline stage: {name}")


def _safe_stage_name(processor_spec: str, processor_cls: Optional[Type[BaseBatchProcessor]] = None) -> str:
    """
    表示・summary用の stage 名を安全に解決する。
    pipeline 正規名が取れればそれを使い、無理なら class 名ベースへフォールバックする。
    """
    try:
        return _normalize_pipeline_stage_name(processor_spec)
    except Exception:
        if processor_cls is not None:
            return processor_cls.__name__
        return str(processor_spec)


def _parse_pipeline_spec(spec: str) -> List[str]:
    """
    `--pipeline` 引数を解析し、stage 名リストへ変換する。
    """
    if spec is None:
        raise ValueError("--pipeline must not be None")

    items = [x.strip() for x in spec.split(",") if x.strip()]
    if not items:
        raise ValueError("--pipeline must not be empty")

    return [_normalize_pipeline_stage_name(x) for x in items]


def _validate_supported_pipeline(stages: List[str]) -> None:
    """
    pipeline がサポート対象かを検証する。
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
# Runtime override normalization
# -------------------------
def _extract_runtime_overrides(exec_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    unknown args から runtime override 候補を抽出する。

    Notes
    -----
    ここで抽出した値は processor 共通の「正規化済み runtime context」。
    実際に各 processor.execute() へ渡すキー名への変換は
    `_build_stage_exec_kwargs()` で行う。
    """
    runtime: Dict[str, Any] = {}

    run_date = None
    if "run_date" in exec_kwargs:
        run_date = exec_kwargs.pop("run_date")
    elif "date" in exec_kwargs:
        run_date = exec_kwargs.pop("date")

    if run_date is not None:
        s = str(run_date).strip()
        if not _DATE_RE.match(s):
            raise ValueError(f"Invalid date format: {s} (expected YYYY-MM-DD)")
        runtime["date"] = s

    target_dir = None
    if "target_dir" in exec_kwargs:
        target_dir = exec_kwargs.pop("target_dir")
    elif "dir" in exec_kwargs:
        target_dir = exec_kwargs.pop("dir")

    if target_dir is not None:
        runtime["target_dir"] = str(target_dir)

    for k in list(exec_kwargs.keys()):
        if k.startswith("nef_"):
            runtime[k] = exec_kwargs.pop(k)

    return runtime


def _get_processor_runtime_param_names(processor_cls: Type[BaseBatchProcessor]) -> set[str]:
    """
    processor が宣言している runtime parameter 名一覧を取得する。
    """
    names = getattr(processor_cls, "runtime_param_names", ()) or ()
    return {str(x) for x in names}


def _build_runtime_candidates_for_stage(
    stage_name: str,
    runtime_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    stage ごとの runtime 候補キーを作る。
    ここでは stage 名差分の吸収だけを行う。
    """
    candidates: Dict[str, Any] = {}

    date_value = runtime_overrides.get("date")
    target_dir = runtime_overrides.get("target_dir")

    if stage_name == "nef":
        if date_value is not None:
            candidates["target_date"] = date_value
        if target_dir is not None:
            candidates["target_dir"] = target_dir

        for k, v in runtime_overrides.items():
            if k.startswith("nef_"):
                candidates[k] = v
    else:
        if date_value is not None:
            candidates["date"] = date_value
        if target_dir is not None:
            candidates["target_dir"] = target_dir

    return candidates


def _build_stage_exec_kwargs(
    processor_spec: str,
    processor_cls: Type[BaseBatchProcessor],
    common_exec_kwargs: Dict[str, Any],
    runtime_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    stage ごとの execute kwargs を構築する。

    Policy
    ------
    - Base FW が apply_runtime_params() で setup 前に適用できる形へ揃える
    - stage ごとの runtime key 差はここで吸収する
    - processor が宣言していない runtime param は渡さない
    - artifact 連携など execute 専用のキーは明示的に残す
    """
    stage_name = _safe_stage_name(processor_spec, processor_cls=processor_cls)
    stage_exec_kwargs = dict(common_exec_kwargs)

    runtime_candidates = _build_runtime_candidates_for_stage(
        stage_name=stage_name,
        runtime_overrides=runtime_overrides,
    )
    allowed_runtime = _get_processor_runtime_param_names(processor_cls)

    # runtime params: processor が宣言したものだけ渡す
    for key, value in runtime_candidates.items():
        if key in allowed_runtime:
            stage_exec_kwargs[key] = value

    # execute 連携で明示的に許可するもの
    passthrough_keys = {
        "input_csv_path",
        "max_images",
    }

    filtered_exec_kwargs: Dict[str, Any] = {}
    for key, value in stage_exec_kwargs.items():
        if key in passthrough_keys or key in allowed_runtime:
            filtered_exec_kwargs[key] = value

    return filtered_exec_kwargs


# -------------------------
# Pipeline artifact helpers
# -------------------------
def _resolve_session_name_from_runtime(
    runtime_overrides: Dict[str, Any],
) -> str:
    """
    runtime override 情報から session 名を解決する。
    """
    target_dir = runtime_overrides.get("target_dir")
    if target_dir:
        return Path(str(target_dir)).name

    date = runtime_overrides.get("date")
    if date:
        return str(date)

    return "ALL"


def _infer_nef_output_csv_path(
    proc: BaseBatchProcessor,
    runtime_overrides: Dict[str, Any],
) -> Optional[str]:
    """
    NEF stage 実行後の output CSV パスを推定する。
    """
    session = _resolve_session_name_from_runtime(runtime_overrides)
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
    """
    candidate_names = (
        "processed_count",
        "processed_count_this_run",
        "images_processed",
        "total_processed",
        "count_processed",
        "success_count",
    )

    for name in candidate_names:
        value = getattr(proc, name, None)
        if isinstance(value, int):
            return value

    processed_images = getattr(proc, "processed_images", None)
    if isinstance(processed_images, (set, list, tuple)):
        return len(processed_images)

    output_data = getattr(proc, "output_data", None)
    if isinstance(output_data, list):
        return len(output_data)

    run_ctx = getattr(proc, "run_ctx", None)
    if run_ctx is not None:
        for name in candidate_names:
            value = getattr(run_ctx, name, None)
            if isinstance(value, int):
                return value

        processed_images = getattr(run_ctx, "processed_images", None)
        if isinstance(processed_images, (set, list, tuple)):
            return len(processed_images)

    return None


def _infer_run_output_dir(proc: BaseBatchProcessor) -> Optional[str]:
    """
    processor インスタンスから run 出力ディレクトリを推定する。
    """
    run_ctx = getattr(proc, "run_ctx", None)
    if run_ctx is not None and getattr(run_ctx, "out_dir", None):
        return str(Path(run_ctx.out_dir))

    project_root = Path(getattr(proc, "project_root", Path.cwd()))
    return str(project_root / "runs" / "latest")


def _build_stage_result(
    processor_spec: str,
    processor_cls: Type[BaseBatchProcessor],
    proc: BaseBatchProcessor,
    runtime_overrides: Dict[str, Any],
    exec_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    stage 実行結果の最小メタ情報を構築する。
    """
    stage_name = _safe_stage_name(processor_spec, processor_cls=processor_cls)

    result: Dict[str, Any] = {
        "name": stage_name,
        "status": "success",
        "input_csv_path": None,
        "output_csv_path": None,
        "processed_count": None,
        "applied_max_images": None,
        "message": None,
        "run_output_dir": _infer_run_output_dir(proc),
    }

    if exec_kwargs is not None:
        result["input_csv_path"] = exec_kwargs.get("input_csv_path")
        result["applied_max_images"] = exec_kwargs.get("max_images")

    result["processed_count"] = _infer_processed_count(proc)

    if stage_name == "nef":
        result["output_csv_path"] = _infer_nef_output_csv_path(proc, runtime_overrides)
        if not result["output_csv_path"]:
            result["status"] = "failed"
            result["message"] = "NEF output CSV path could not be resolved"
    elif stage_name == "portrait_quality":
        result["input_csv_path"] = result["input_csv_path"] or getattr(proc, "input_csv_path", None)

    return result


def _build_pipeline_run_context(
    ctor_kwargs: Dict[str, Any],
    exec_kwargs: Dict[str, Any],
    runtime_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    pipeline summary 用の runtime context を構築する。
    """
    config_paths = ctor_kwargs.get("config_paths")
    if isinstance(config_paths, tuple):
        config_paths = list(config_paths)

    return {
        "date": runtime_overrides.get("date"),
        "target_dir": runtime_overrides.get("target_dir"),
        "config_path": ctor_kwargs.get("config_path"),
        "max_workers": ctor_kwargs.get("max_workers"),
        "max_images": exec_kwargs.get("max_images"),
        "config_env": ctor_kwargs.get("config_env"),
        "config_paths": config_paths,
    }


def _derive_pipeline_status(stage_results: List[Dict[str, Any]]) -> str:
    """
    stage 実行結果から pipeline 全体の status を集約する。
    """
    if not stage_results:
        return "failed"

    for stage in stage_results:
        if stage.get("status") != "success":
            return "failed"

    return "success"


def _build_pipeline_summary(
    stages: List[str],
    stage_results: List[Dict[str, Any]],
    ctor_kwargs: Dict[str, Any],
    exec_kwargs: Dict[str, Any],
    runtime_overrides: Dict[str, Any],
    duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """
    pipeline 実行結果から summary を構築する。
    """
    summary = {
        "summary_version": 1,
        "pipeline": stages,
        "status": _derive_pipeline_status(stage_results),
        "run_context": _build_pipeline_run_context(
            ctor_kwargs=ctor_kwargs,
            exec_kwargs=exec_kwargs,
            runtime_overrides=runtime_overrides,
        ),
        "stages": stage_results,
    }
    if duration_sec is not None:
        summary["duration_sec"] = round(float(duration_sec), 3)
    return summary


def _resolve_pipeline_summary_output_path(summary: Dict[str, Any]) -> Path:
    """
    pipeline summary の出力先パスを解決する。
    """
    stages = summary.get("stages", [])
    if stages:
        first_run_output_dir = stages[0].get("run_output_dir")
        if first_run_output_dir:
            return Path(first_run_output_dir) / "pipeline_summary.json"

    cwd_runs_latest = Path.cwd() / "runs" / "latest"
    return cwd_runs_latest / "pipeline_summary.json"


def _write_pipeline_summary_json(summary: Dict[str, Any]) -> Path:
    """
    pipeline summary を JSON ファイルとして保存する。
    """
    output_path = _resolve_pipeline_summary_output_path(summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _print_pipeline_summary(summary: Dict[str, Any]) -> None:
    """
    pipeline summary を標準出力へ表示する。
    """
    print("[pipeline summary]")
    print(f"pipeline = {','.join(summary['pipeline'])}")
    print(f"status = {summary['status']}")

    if summary.get("duration_sec") is not None:
        print(f"duration_sec = {summary['duration_sec']}")

    run_context = summary.get("run_context")
    if run_context:
        if run_context.get("date") is not None:
            print(f"date = {run_context['date']}")
        if run_context.get("target_dir") is not None:
            print(f"target_dir = {run_context['target_dir']}")
        if run_context.get("max_images") is not None:
            print(f"max_images = {run_context['max_images']}")

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
    runtime_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    単一 processor を実行する。

    Notes
    -----
    runtime override は Base FW が execute(**kwargs) 内で吸収する。
    """
    processor_cls = resolve_processor(processor_spec)
    proc = processor_cls(**ctor_kwargs)
    proc.execute(**exec_kwargs)
    return _build_stage_result(
        processor_spec=processor_spec,
        processor_cls=processor_cls,
        proc=proc,
        runtime_overrides=runtime_overrides,
        exec_kwargs=exec_kwargs,
    )


def run_pipeline_chain(
    stages: List[str],
    ctor_kwargs: Dict[str, Any],
    exec_kwargs: Dict[str, Any],
    runtime_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    pipeline chain を順次実行する。

    Notes
    -----
    - max_images は pipeline 先頭 stage のみに適用
    - 後続 stage は upstream artifact に従う
    """
    previous_result: Optional[Dict[str, Any]] = None
    stage_results: List[Dict[str, Any]] = []
    started_at = time.time()

    for idx, stage_name in enumerate(stages):
        processor_cls = resolve_processor(stage_name)
        stage_exec_kwargs = _build_stage_exec_kwargs(
            processor_spec=stage_name,
            processor_cls=processor_cls,
            common_exec_kwargs=exec_kwargs,
            runtime_overrides=runtime_overrides,
        )

        # max_images は pipeline 先頭 stage にのみ適用
        if idx > 0:
            stage_exec_kwargs.pop("max_images", None)

        if stage_name == "portrait_quality":
            if previous_result is None or previous_result.get("name") != "nef":
                raise RuntimeError("portrait_quality stage requires a previous nef stage result")

            if previous_result.get("status") != "success":
                raise RuntimeError("portrait_quality stage requires a successful nef stage result")

            nef_csv = previous_result.get("output_csv_path")
            if not nef_csv:
                raise FileNotFoundError("NEF output CSV path could not be resolved after nef stage execution")

            stage_exec_kwargs["input_csv_path"] = nef_csv

        previous_result = run_single_processor(
            processor_spec=stage_name,
            ctor_kwargs=dict(ctor_kwargs),
            exec_kwargs=stage_exec_kwargs,
            runtime_overrides=dict(runtime_overrides),
        )
        stage_results.append(previous_result)

        if previous_result.get("status") != "success":
            break

    duration_sec = time.time() - started_at

    return _build_pipeline_summary(
        stages=stages,
        stage_results=stage_results,
        ctor_kwargs=ctor_kwargs,
        exec_kwargs=exec_kwargs,
        runtime_overrides=runtime_overrides,
        duration_sec=duration_sec,
    )


# -------------------------
# CLI
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    """
    CLI 引数パーサを生成する。
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
    p.add_argument(
        "--config-env",
        default=None,
        help="ConfigManager env name (optional)",
    )
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


def _validate_entrypoint_args(args: argparse.Namespace) -> None:
    """
    CLI 引数の整合性を検証する。
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
    runtime_overrides = _extract_runtime_overrides(exec_kwargs)

    if args.pipeline:
        stages = _parse_pipeline_spec(args.pipeline)
        _validate_supported_pipeline(stages)

        if args.dry_run:
            print(f"[dry-run] pipeline = {stages}")
            print(f"[dry-run] ctor_kwargs = {ctor_kwargs}")
            print(f"[dry-run] exec_kwargs = {exec_kwargs}")
            print(f"[dry-run] runtime_overrides = {runtime_overrides}")
            for stage in stages:
                processor_cls = resolve_processor(stage)
                print(
                    f"[dry-run] stage_exec_kwargs[{stage}] = "
                    f"{_build_stage_exec_kwargs(stage, processor_cls, exec_kwargs, runtime_overrides)}"
                )
            return 0

        summary = run_pipeline_chain(
            stages=stages,
            ctor_kwargs=ctor_kwargs,
            exec_kwargs=exec_kwargs,
            runtime_overrides=runtime_overrides,
        )
        _print_pipeline_summary(summary)
        summary_path = _write_pipeline_summary_json(summary)
        print(f"pipeline_summary_json = {summary_path}")
        return 0

    assert args.processor is not None

    processor_cls = resolve_processor(args.processor)

    if args.dry_run:
        print(f"[dry-run] processor = {processor_cls.__module__}.{processor_cls.__name__}")
        print(f"[dry-run] ctor_kwargs = {ctor_kwargs}")
        print(f"[dry-run] exec_kwargs = {exec_kwargs}")
        print(f"[dry-run] runtime_overrides = {runtime_overrides}")
        print(
            f"[dry-run] stage_exec_kwargs = "
            f"{_build_stage_exec_kwargs(args.processor, processor_cls, exec_kwargs, runtime_overrides)}"
        )
        return 0

    stage_exec_kwargs = _build_stage_exec_kwargs(
        processor_spec=args.processor,
        processor_cls=processor_cls,
        common_exec_kwargs=exec_kwargs,
        runtime_overrides=runtime_overrides,
    )

    run_single_processor(
        processor_spec=args.processor,
        ctor_kwargs=ctor_kwargs,
        exec_kwargs=stage_exec_kwargs,
        runtime_overrides=runtime_overrides,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
