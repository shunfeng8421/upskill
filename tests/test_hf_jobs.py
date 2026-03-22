from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

import upskill.hf_jobs as hf_jobs
from upskill.hf_jobs import (
    JobsConfig,
    SubmittedJob,
    _make_run_id,
    _normalize_job_id,
    _submit_bundle_job,
    parse_duration_seconds,
    wait_for_job_outputs,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_duration_seconds_supports_hf_style_suffixes() -> None:
    assert parse_duration_seconds("45m") == 2700.0
    assert parse_duration_seconds("2h") == 7200.0
    assert parse_duration_seconds("30") == 30.0


def test_make_run_id_adds_entropy_even_with_frozen_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: object | None = None) -> FrozenDateTime:
            del tz
            return cls(2026, 3, 22, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr("upskill.hf_jobs.datetime", FrozenDateTime)

    run_id_a = _make_run_id("with-skill", "qwen35", "pull-request-descriptions")
    run_id_b = _make_run_id("with-skill", "qwen35", "pull-request-descriptions")

    assert run_id_a != run_id_b
    assert run_id_a.startswith("20260322T120000Z_with-skill-qwen35-pull-request-descriptions_")
    assert run_id_b.startswith("20260322T120000Z_with-skill-qwen35-pull-request-descriptions_")


def test_normalize_job_id_extracts_namespace_and_id_from_url() -> None:
    assert (
        _normalize_job_id("View at: https://huggingface.co/jobs/evalstate/69bd5e5f71691dc46f161e83")
        == "evalstate/69bd5e5f71691dc46f161e83"
    )


def test_wait_for_job_outputs_downloads_full_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    messages: list[str] = []

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
        progress_callback=messages.append,
    )

    assert output_dir == tmp_path / "outputs" / "run-456"
    assert len(calls) == 3
    assert messages[0] == "waiting for job job-123 (run_id=run-456)"
    assert "completed; downloading artifacts" in messages[1]


def test_wait_for_job_outputs_raises_when_job_enters_error_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        if args[:4] == ["hf", "jobs", "ps", "-a"]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout=(
                    '[{"id":"job-123","owner":{"name":"evalstate"},'
                    '"status":{"stage":"ERROR","message":"boom"}}]'
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ended with stage ERROR"):
        wait_for_job_outputs(
            SubmittedJob(job_id="evalstate/job-123", run_id="run-456", artifact_repo="ns/repo"),
            destination_root=tmp_path,
            wait_timeout_seconds=1.0,
            poll_interval_seconds=0.01,
        )


def test_submit_bundle_job_retries_conflict_upload_and_auth_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    bundle_archive = tmp_path / "bundle.tar.gz"
    bundle_archive.write_text("bundle", encoding="utf-8")

    def fake_run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(args)
        if args[:2] == ["hf", "download"] and "--dry-run" in args:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        if args[:2] == ["hf", "upload"] and args[4].endswith("bundle.tar.gz"):
            upload_attempt = sum(
                1
                for call in calls
                if call[:2] == ["hf", "upload"] and call[4].endswith("bundle.tar.gz")
            )
            if upload_attempt == 1:
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=1,
                    stdout="",
                    stderr="412 Precondition Failed\nA commit has happened since. Please refresh and try again.\n",
                )
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        if args[:3] == ["hf", "jobs", "run"]:
            submit_attempt = sum(1 for call in calls if call[:3] == ["hf", "jobs", "run"])
            if submit_attempt == 1:
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=1,
                    stdout="Set HF_DEBUG=1 as environment variable for full traceback.\n",
                    stderr=(
                        "Error: You've hit the rate limit for the /whoami-v2 endpoint, "
                        "which is intentionally strict for security reasons.\n"
                    ),
                )
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="View at: https://huggingface.co/jobs/evalstate/job-123\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("upskill.hf_jobs.time.sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("upskill.hf_jobs._VERIFIED_ARTIFACT_REPOS", set())

    submission = _submit_bundle_job(
        bundle_archive=bundle_archive,
        jobs_config=JobsConfig(artifact_repo="ns/repo"),
        run_id="run-456",
        model="qwen35",
    )

    assert submission == SubmittedJob(
        job_id="evalstate/job-123",
        run_id="run-456",
        artifact_repo="ns/repo",
    )
    assert sum(1 for call in calls if call[:2] == ["hf", "download"] and "--dry-run" in call) == 1
    assert sum(1 for call in calls if call[:2] == ["hf", "upload"]) == 2
    assert sum(1 for call in calls if call[:3] == ["hf", "jobs", "run"]) == 2


def test_submit_bundle_job_checks_artifact_repo_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    bundle_archive = tmp_path / "bundle.tar.gz"
    bundle_archive.write_text("bundle", encoding="utf-8")

    def fake_run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(args)
        if args[:2] == ["hf", "download"] and "--dry-run" in args:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        if args[:2] == ["hf", "upload"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        if args[:3] == ["hf", "jobs", "run"]:
            run_number = sum(1 for call in calls if call[:3] == ["hf", "jobs", "run"])
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout=f"View at: https://huggingface.co/jobs/evalstate/job-{run_number}\n",
                stderr="",
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("upskill.hf_jobs._VERIFIED_ARTIFACT_REPOS", set())

    first = _submit_bundle_job(
        bundle_archive=bundle_archive,
        jobs_config=JobsConfig(artifact_repo="ns/repo"),
        run_id="run-1",
        model="qwen35",
    )
    second = _submit_bundle_job(
        bundle_archive=bundle_archive,
        jobs_config=JobsConfig(artifact_repo="ns/repo"),
        run_id="run-2",
        model="qwen35",
    )

    assert first.job_id == "evalstate/job-1"
    assert second.job_id == "evalstate/job-2"
    assert sum(1 for call in calls if call[:2] == ["hf", "download"] and "--dry-run" in call) == 1
    assert {"ns/repo"} == hf_jobs._VERIFIED_ARTIFACT_REPOS
    jobs_run_call = calls[-1]
    assert "ghcr.io/astral-sh/uv:python3.13-bookworm" in jobs_run_call
    assert any("huggingface_hub[cli]==1.7.2" in arg for arg in jobs_run_call)
