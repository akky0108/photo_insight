from __future__ import annotations

import pytest

from photo_insight.cli import run_batch


def test_validate_entrypoint_args_rejects_processor_and_pipeline() -> None:
    parser = run_batch.build_parser()
    args = parser.parse_args(["--processor", "nef", "--pipeline", "nef,portrait_quality"])

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_batch._validate_entrypoint_args(args)


def test_validate_entrypoint_args_requires_processor_or_pipeline() -> None:
    parser = run_batch.build_parser()
    args = parser.parse_args([])

    with pytest.raises(
        ValueError,
        match="Either --processor or --pipeline must be specified",
    ):
        run_batch._validate_entrypoint_args(args)


def test_parse_pipeline_spec_normalizes_supported_names() -> None:
    stages = run_batch._parse_pipeline_spec("nef,portrait_quality")
    assert stages == ["nef", "portrait_quality"]


def test_parse_pipeline_spec_normalizes_aliases() -> None:
    stages = run_batch._parse_pipeline_spec("nef_file_batch,portrait")
    assert stages == ["nef", "portrait_quality"]


def test_validate_supported_pipeline_accepts_nef_portrait_quality() -> None:
    stages = ["nef", "portrait_quality"]
    run_batch._validate_supported_pipeline(stages)


def test_validate_supported_pipeline_rejects_unsupported_combination() -> None:
    with pytest.raises(ValueError, match="Unsupported pipeline"):
        run_batch._validate_supported_pipeline(["nef", "evaluation_rank"])


def test_main_pipeline_dry_run_returns_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = run_batch.main(
        [
            "--pipeline",
            "nef,portrait_quality",
            "--dry-run",
            "--date",
            "2026-02-17",
            "--target-dir",
            "/work/input/2026-02-17",
        ]
    )

    assert rc == 0

    out = capsys.readouterr().out
    assert "[dry-run] pipeline = ['nef', 'portrait_quality']" in out
    assert "[dry-run] exec_kwargs = {}" in out
    assert "'date': '2026-02-17'" in out
    assert "'target_dir': '/work/input/2026-02-17'" in out


def test_main_pipeline_dry_run_rejects_unsupported_pipeline() -> None:
    with pytest.raises(ValueError, match="Unsupported pipeline"):
        run_batch.main(
            [
                "--pipeline",
                "nef,evaluation_rank",
                "--dry-run",
            ]
        )


def test_main_single_processor_dry_run_still_works(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = run_batch.main(
        [
            "--processor",
            "nef",
            "--dry-run",
            "--date",
            "2026-02-17",
            "--target-dir",
            "/work/input/2026-02-17",
        ]
    )

    assert rc == 0

    out = capsys.readouterr().out
    assert "[dry-run] processor =" in out
    assert "NEFFileBatchProcess" in out
    assert "[dry-run] exec_kwargs = {}" in out
    assert "'date': '2026-02-17'" in out
    assert "'target_dir': '/work/input/2026-02-17'" in out


def test_parse_unknown_args_rejects_reserved_pipeline_key() -> None:
    with pytest.raises(ValueError, match="reserved runner/CLI option"):
        run_batch._parse_unknown_args(["--pipeline", "nef,portrait_quality"])


def test_extract_runtime_overrides_moves_date_and_target_dir() -> None:
    exec_kwargs = {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026-02-17",
        "append_mode": True,
    }

    injected = run_batch._extract_runtime_overrides(exec_kwargs)

    assert injected == {
        "date": "2026-02-17",
        "target_dir": "/work/input/2026-02-17",
    }
    assert exec_kwargs == {
        "append_mode": True,
    }


def test_normalize_pipeline_stage_name_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        run_batch._normalize_pipeline_stage_name("unknown_stage")


def test_parse_pipeline_spec_rejects_empty_spec() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        run_batch._parse_pipeline_spec(" , , ")
