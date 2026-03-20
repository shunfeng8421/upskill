from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from upskill.hf_jobs import (
    JobsConfig,
    SubmittedJob,
    build_submit_eval_job_command,
    parse_duration_seconds,
    submit_eval_job,
    wait_for_job_outputs,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_parse_duration_seconds_supports_hf_style_suffixes() -> None:
    assert parse_duration_seconds("45m") == 2700.0
    assert parse_duration_seconds("2h") == 7200.0
    assert parse_duration_seconds("30") == 30.0


def test_build_submit_eval_job_command_includes_noninteractive_flags(tmp_path: Path) -> None:
    command = build_submit_eval_job_command(
        skill_dir=tmp_path / "skill",
        config=JobsConfig(
            artifact_repo="namespace/upskill-evals",
            wait=True,
            jobs_timeout="45m",
            jobs_flavor="cpu-basic",
            jobs_secrets="HF_TOKEN,OPENROUTER_API_KEY",
            jobs_namespace="my-org",
        ),
        models=["qwen35"],
        runs=1,
        no_baseline=True,
        verbose=True,
    )

    assert "--artifact-repo" in command
    assert "--skill-dir" in command
    assert "--models" in command
    assert "--no-baseline" in command
    assert "--verbose" in command
    assert "--yes" in command
    assert "--json" in command
    assert "--namespace" in command


def test_submit_eval_job_parses_json_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"job_id":"job-123","run_id":"run-456","artifact_repo":"ns/repo"}\n',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    submission = submit_eval_job(
        skill_dir=tmp_path / "skill",
        config=JobsConfig(artifact_repo="ns/repo"),
        models=["qwen35"],
        runs=1,
        no_baseline=False,
        verbose=False,
    )

    assert submission.job_id == "job-123"
    assert submission.run_id == "run-456"
    assert submission.artifact_repo == "ns/repo"


def test_wait_for_job_outputs_downloads_full_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(args)
        if args[2] == "ns/repo" and args[3].endswith("exit_code.txt"):
            marker = tmp_path / "outputs" / "run-456" / "exit_code.txt"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("0\n", encoding="utf-8")
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        output_dir = tmp_path / "outputs" / "run-456"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "exit_code.txt").write_text("0\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output_dir = wait_for_job_outputs(
        SubmittedJob(job_id="job-123", run_id="run-456", artifact_repo="ns/repo"),
        destination_root=tmp_path,
        wait_timeout_seconds=1.0,
        poll_interval_seconds=0.01,
    )

    assert output_dir == tmp_path / "outputs" / "run-456"
    assert len(calls) == 2
