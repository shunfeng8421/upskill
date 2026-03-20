from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from upskill.hf_jobs import (
    JobsConfig,
    SubmittedJob,
    _normalize_job_id,
    _parse_submission_payload,
    build_submit_eval_job_command,
    parse_duration_seconds,
    run_remote_eval,
    submit_eval_job,
    wait_for_job_outputs,
)
from upskill.models import ExpectedSpec, Skill, TestCase

if TYPE_CHECKING:
    from pathlib import Path


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


def test_parse_submission_payload_handles_preamble_lines() -> None:
    payload = _parse_submission_payload(
        "RUN_ID=abc123\n"
        "Secrets to forward:\n"
        "  - HF_TOKEN (present locally)\n"
        '{"job_id":"job-123","run_id":"run-456","artifact_repo":"ns/repo"}\n'
    )

    assert payload["job_id"] == "job-123"


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


def test_run_remote_eval_submits_with_skill_and_baseline_before_waiting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    skill = Skill(
        name="pull-request-descriptions",
        description="Write good pull request descriptions.",
        body="Use a clear structure.",
    )
    test_cases = [TestCase(input="prompt", expected=ExpectedSpec(contains=["answer"]))]

    def fake_submit_fast_agent_eval_job(**kwargs: object) -> SubmittedJob:
        phase_label = kwargs["phase_label"]
        events.append(f"submit:{phase_label}")
        return SubmittedJob(
            job_id=f"evalstate/{phase_label}-job",
            run_id=f"{phase_label}-run",
            artifact_repo="ns/repo",
        )

    def fake_wait_for_job_outputs(
        job: SubmittedJob,
        *,
        destination_root: Path,
        wait_timeout_seconds: float,
        progress_callback: object = None,
    ) -> Path:
        del wait_timeout_seconds, progress_callback
        events.append(f"wait:{job.run_id}")
        output_dir = destination_root / "outputs" / job.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = output_dir / "results"
        status_dir = output_dir / "status"
        results_dir.mkdir()
        status_dir.mkdir()
        result_file = results_dir / "test_1.json"
        result_file.write_text(
            '{"messages":[{"role":"assistant","content":[{"type":"text","text":"answer"}]}]}',
            encoding="utf-8",
        )
        (status_dir / "test_1.exit_code.txt").write_text("0\n", encoding="utf-8")
        return output_dir

    monkeypatch.setattr(
        "upskill.hf_jobs.submit_fast_agent_eval_job", fake_submit_fast_agent_eval_job
    )
    monkeypatch.setattr("upskill.hf_jobs.wait_for_job_outputs", fake_wait_for_job_outputs)

    results, job_refs = run_remote_eval(
        skill=skill,
        test_cases=test_cases,
        model="qwen35",
        jobs_config=JobsConfig(artifact_repo="ns/repo", wait=True),
        fastagent_config_path=tmp_path / "fastagent.config.yaml",
        destination_root=tmp_path / "remote",
        run_baseline=True,
    )

    assert results is not None
    assert job_refs == ["evalstate/with-skill-job", "evalstate/baseline-job"]
    assert events[:2] == ["submit:with-skill", "submit:baseline"]
    assert sorted(events[2:]) == ["wait:baseline-run", "wait:with-skill-run"]
