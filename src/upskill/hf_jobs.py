"""Helpers for submitting and collecting Hugging Face Jobs-based eval runs."""

from __future__ import annotations

import concurrent.futures
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from upskill.artifacts import copy_config_file, ensure_directory, materialize_workspace
from upskill.evaluate import apply_eval_metrics, check_expected, format_test_prompt
from upskill.models import EvalResults, Skill, TestCase, TestResult
from upskill.result_parsing import parse_fast_agent_results

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class JobsConfig:
    """Configuration for remote Jobs-backed execution."""

    artifact_repo: str
    wait: bool = False
    jobs_timeout: str = "2h"
    jobs_flavor: str = "cpu-basic"
    jobs_secrets: str = "HF_TOKEN"
    jobs_namespace: str | None = None


@dataclass(frozen=True)
class SubmittedJob:
    """A submitted Hugging Face Job plus its artifact identifiers."""

    job_id: str
    run_id: str
    artifact_repo: str


@dataclass(frozen=True)
class RemoteEvalJobResult:
    """Downloaded artifact directory plus submitted job info."""

    job: SubmittedJob
    output_dir: Path


_JOB_URL_RE = re.compile(r"https://huggingface\.co/jobs/(?P<namespace>[^/]+)/(?P<job_id>[^/\s]+)")


def _normalize_job_id(value: str) -> str:
    """Normalize a raw job reference into ``job_id`` or ``namespace/job_id`` form."""
    raw = value.strip()
    match = _JOB_URL_RE.search(raw)
    if match:
        return f"{match.group('namespace')}/{match.group('job_id')}"
    if raw.startswith("View at:"):
        return _normalize_job_id(raw.removeprefix("View at:"))
    return raw


def _split_job_reference(value: str) -> tuple[str | None, str]:
    normalized = _normalize_job_id(value)
    if "/" in normalized:
        namespace, job_id = normalized.rsplit("/", 1)
        return namespace, job_id
    return None, normalized


def _parse_submission_payload(stdout: str) -> dict[str, object]:
    """Parse the JSON payload from submission script output.

    The wrapper scripts may print human-readable preamble lines before the final JSON line.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"Unexpected HF eval submission output: {stdout}")


def _lookup_job_stage(job_id: str) -> str | None:
    """Best-effort lookup of an HF job stage from ``hf jobs ps --format json``."""
    completed = subprocess.run(
        ["hf", "jobs", "ps", "-a", "--format", "json"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None

    namespace, bare_job_id = _split_job_reference(job_id)
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("id", "")) != bare_job_id:
            continue
        owner = entry.get("owner")
        owner_name = owner.get("name") if isinstance(owner, dict) else None
        if namespace is not None and owner_name != namespace:
            continue
        status = entry.get("status")
        if isinstance(status, dict):
            stage = status.get("stage")
            if isinstance(stage, str):
                return stage
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_duration_seconds(value: str) -> float:
    """Parse a simple HF-style duration like ``45m`` or ``2h``."""
    if not value:
        raise ValueError("Duration value must not be empty.")
    suffix = value[-1]
    multiplier = {
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }.get(suffix)
    if multiplier is None:
        suffix = "s"
        multiplier = 1.0
        number = value
    else:
        number = value[:-1]
    try:
        return float(number) * multiplier
    except ValueError as exc:
        raise ValueError(f"Invalid duration value: {value}") from exc


def _sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return sanitized or "eval"


def _make_run_id(*parts: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "-".join(_sanitize_label(part) for part in parts if part)
    return f"{timestamp}_{suffix}" if suffix else timestamp


def build_submit_eval_job_command(
    *,
    skill_dir: Path,
    config: JobsConfig,
    models: list[str],
    runs: int,
    no_baseline: bool,
    verbose: bool,
    tests_path: Path | None = None,
) -> list[str]:
    """Build the repository wrapper command for submitting a remote eval job."""
    command = [
        str(_repo_root() / "scripts" / "hf" / "submit_hf_eval_job.sh"),
        "--artifact-repo",
        config.artifact_repo,
        "--skill-dir",
        str(skill_dir),
        "--models",
        ",".join(models),
        "--runs",
        str(runs),
        "--flavor",
        config.jobs_flavor,
        "--timeout",
        config.jobs_timeout,
        "--secrets",
        config.jobs_secrets,
        "--yes",
        "--json",
    ]
    if tests_path is not None:
        command.extend(["--tests", str(tests_path)])
    if no_baseline:
        command.append("--no-baseline")
    if verbose:
        command.append("--verbose")
    if config.jobs_namespace:
        command.extend(["--namespace", config.jobs_namespace])
    return command


def submit_eval_job(
    *,
    skill_dir: Path,
    config: JobsConfig,
    models: list[str],
    runs: int,
    no_baseline: bool,
    verbose: bool,
    tests_path: Path | None = None,
) -> SubmittedJob:
    """Submit an eval job via the checked-in HF wrapper script."""
    command = build_submit_eval_job_command(
        skill_dir=skill_dir,
        config=config,
        models=models,
        runs=runs,
        no_baseline=no_baseline,
        verbose=verbose,
        tests_path=tests_path,
    )
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to submit HF eval job:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    payload = _parse_submission_payload(completed.stdout)
    return SubmittedJob(
        job_id=_normalize_job_id(str(payload["job_id"])),
        run_id=str(payload["run_id"]),
        artifact_repo=str(payload["artifact_repo"]),
    )


def wait_for_job_outputs(
    job: SubmittedJob,
    *,
    destination_root: Path,
    wait_timeout_seconds: float,
    poll_interval_seconds: float = 15.0,
    progress_callback: Callable[[str], None] | None = None,
) -> Path:
    """Wait until a job uploads its exit marker, then download full outputs."""
    deadline = time.monotonic() + wait_timeout_seconds
    marker_path = f"outputs/{job.run_id}/exit_code.txt"
    poll_count = 0

    if progress_callback is not None:
        progress_callback(f"waiting for job {job.job_id} (run_id={job.run_id})")

    while time.monotonic() < deadline:
        poll_count += 1
        stage = _lookup_job_stage(job.job_id)
        marker_download = subprocess.run(
            [
                "hf",
                "download",
                job.artifact_repo,
                marker_path,
                "--repo-type",
                "dataset",
                "--local-dir",
                str(destination_root),
                "--quiet",
            ],
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
        )
        if marker_download.returncode == 0:
            if progress_callback is not None:
                progress_callback(f"job {job.job_id} completed; downloading artifacts")
            full_download = subprocess.run(
                [
                    "hf",
                    "download",
                    job.artifact_repo,
                    "--repo-type",
                    "dataset",
                    "--include",
                    f"outputs/{job.run_id}/**",
                    "--local-dir",
                    str(destination_root),
                ],
                cwd=_repo_root(),
                check=False,
                capture_output=True,
                text=True,
            )
            if full_download.returncode != 0:
                raise RuntimeError(
                    "HF job finished but artifacts could not be downloaded:\n"
                    f"stdout:\n{full_download.stdout}\n"
                    f"stderr:\n{full_download.stderr}"
                )
            if progress_callback is not None:
                progress_callback(f"downloaded artifacts for job {job.job_id}")
            return destination_root / "outputs" / job.run_id
        if stage in {"ERROR", "CANCELED", "DELETED"}:
            raise RuntimeError(
                f"HF job {job.job_id} ended with stage {stage}. "
                f"Inspect logs with `hf jobs logs {job.job_id}`."
            )
        if progress_callback is not None:
            stage_suffix = f" ({stage.lower()})" if stage else ""
            progress_callback(f"poll {poll_count}: job {job.job_id} still running{stage_suffix}")
        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Timed out waiting for HF job artifacts for job {job.job_id} (run_id={job.run_id})."
    )


def _hf_secret_flags(secrets: str) -> list[str]:
    flags: list[str] = []
    for secret in (item.strip() for item in secrets.split(",")):
        if not secret:
            continue
        flags.extend(["--secrets", secret])
    return flags


def _copy_request_bundle(
    *,
    bundle_root: Path,
    test_cases: list[TestCase],
    fastagent_config_path: Path,
    skill: Skill | None,
) -> None:
    """Materialize a remote fast-agent batch bundle."""
    ensure_directory(bundle_root)
    skills_dir = ensure_directory(bundle_root / "skills")
    if skill is not None:
        skill.save(skills_dir / skill.name, tests=test_cases)
    copy_config_file(fastagent_config_path.resolve(), bundle_root / "fastagent.config.yaml")

    requests_root = ensure_directory(bundle_root / "requests")
    manifest_requests: list[dict[str, object]] = []
    for index, test_case in enumerate(test_cases, start=1):
        request_id = f"test_{index}"
        request_dir = ensure_directory(requests_root / request_id)
        prompt_path = request_dir / "prompt.txt"
        prompt_path.write_text(format_test_prompt(test_case), encoding="utf-8")
        workspace_dir = ensure_directory(request_dir / "workspace")
        workspace_files = (
            dict(test_case.context.files) if test_case.context and test_case.context.files else {}
        )
        materialize_workspace(workspace_dir, workspace_files)
        shell_enabled = bool(workspace_files or test_case.validator or test_case.output_file)
        (request_dir / "enable_shell.txt").write_text(
            "1" if shell_enabled else "0", encoding="utf-8"
        )
        manifest_requests.append(
            {
                "id": request_id,
                "index": index,
                "has_workspace_files": bool(workspace_files),
                "enable_shell": shell_enabled,
            }
        )

    (bundle_root / "manifest.json").write_text(
        json.dumps(
            {
                "request_count": len(test_cases),
                "requests": manifest_requests,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _create_bundle_archive(
    *,
    skill: Skill | None,
    test_cases: list[TestCase],
    fastagent_config_path: Path,
) -> tuple[Path, Path]:
    """Create a tar.gz bundle for remote fast-agent evaluation."""
    temp_root = Path(tempfile.mkdtemp(prefix="upskill_hf_eval_"))
    bundle_root = temp_root / "bundle"
    _copy_request_bundle(
        bundle_root=bundle_root,
        test_cases=test_cases,
        fastagent_config_path=fastagent_config_path,
        skill=skill,
    )
    entrypoint_source = _repo_root() / "scripts" / "hf" / "job_entrypoint_eval_fast_agent.sh"
    shutil.copy2(entrypoint_source, bundle_root / "job_entrypoint.sh")
    bundle_archive = temp_root / "bundle.tar.gz"
    with tarfile.open(bundle_archive, "w:gz") as archive:
        archive.add(bundle_root, arcname="bundle")
    return temp_root, bundle_archive


def _submit_bundle_job(
    *,
    bundle_archive: Path,
    jobs_config: JobsConfig,
    run_id: str,
    model: str,
) -> SubmittedJob:
    subprocess.run(
        ["hf", "repo", "create", jobs_config.artifact_repo, "--repo-type", "dataset", "--exist-ok"],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    upload = subprocess.run(
        [
            "hf",
            "upload",
            jobs_config.artifact_repo,
            str(bundle_archive),
            f"inputs/{run_id}/bundle.tar.gz",
            "--repo-type",
            "dataset",
            "--commit-message",
            f"inputs: {run_id}",
        ],
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if upload.returncode != 0:
        raise RuntimeError(
            "Failed to upload remote eval bundle:\n"
            f"stdout:\n{upload.stdout}\n"
            f"stderr:\n{upload.stderr}"
        )

    job_cmd = (
        "set -euo pipefail\n"
        "WORK=/workspace\n"
        'mkdir -p "$WORK/out"\n'
        'cd "$WORK"\n'
        'uv pip install --system "huggingface_hub[cli]>=1.0" "fast-agent-mcp==0.6.2"\n'
        'hf download "$ARTIFACT_REPO" "inputs/$RUN_ID/bundle.tar.gz" --repo-type dataset --local-dir "$WORK"\n'
        'tar -xzf "$WORK/inputs/$RUN_ID/bundle.tar.gz" -C "$WORK"\n'
        "set +e\n"
        'bash "$WORK/bundle/job_entrypoint.sh" "$WORK/bundle" "$WORK/out"\n'
        "status=$?\n"
        "set -e\n"
        'echo "$status" > "$WORK/out/exit_code.txt"\n'
        'hf upload "$ARTIFACT_REPO" "$WORK/out" "outputs/$RUN_ID" --repo-type dataset '
        '--commit-message "outputs: $RUN_ID (exit=$status)"\n'
        'exit "$status"\n'
    )
    command = [
        "hf",
        "jobs",
        "run",
        "--detach",
        "--flavor",
        jobs_config.jobs_flavor,
        "--timeout",
        jobs_config.jobs_timeout,
        *_hf_secret_flags(jobs_config.jobs_secrets),
        "--env",
        f"ARTIFACT_REPO={jobs_config.artifact_repo}",
        "--env",
        f"RUN_ID={run_id}",
        "--env",
        f"FAST_MODEL={model}",
    ]
    if jobs_config.jobs_namespace:
        command.extend(["--namespace", jobs_config.jobs_namespace])
    command.extend(
        [
            "--",
            "ghcr.io/astral-sh/uv:python3.13-bookworm",
            "bash",
            "-lc",
            job_cmd,
        ]
    )
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to submit remote fast-agent eval job:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    job_ref = _normalize_job_id(completed.stdout.strip().splitlines()[-1])
    return SubmittedJob(job_id=job_ref, run_id=run_id, artifact_repo=jobs_config.artifact_repo)


def submit_fast_agent_eval_job(
    *,
    skill: Skill | None,
    test_cases: list[TestCase],
    fastagent_config_path: Path,
    model: str,
    jobs_config: JobsConfig,
    destination_root: Path,
    phase_label: str,
    progress_callback: Callable[[str], None] | None = None,
) -> RemoteEvalJobResult | SubmittedJob:
    """Submit a remote fast-agent evaluation batch and optionally collect artifacts."""
    temp_root, bundle_archive = _create_bundle_archive(
        skill=skill,
        test_cases=test_cases,
        fastagent_config_path=fastagent_config_path,
    )
    try:
        run_id = _make_run_id(phase_label, model, skill.name if skill else "baseline")
        submission = _submit_bundle_job(
            bundle_archive=bundle_archive,
            jobs_config=jobs_config,
            run_id=run_id,
            model=model,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    if progress_callback is not None:
        progress_callback(
            f"submitted remote {phase_label} batch as job {submission.job_id} (run_id={submission.run_id})"
        )

    if not jobs_config.wait:
        return submission

    output_dir = wait_for_job_outputs(
        submission,
        destination_root=destination_root,
        wait_timeout_seconds=parse_duration_seconds(jobs_config.jobs_timeout),
        progress_callback=progress_callback,
    )
    return RemoteEvalJobResult(job=submission, output_dir=output_dir)


def reconstruct_remote_eval_results(
    *,
    output_dir: Path,
    test_cases: list[TestCase],
) -> list[TestResult]:
    """Reconstruct per-test results from a downloaded remote fast-agent batch."""
    results: list[TestResult] = []
    for index, test_case in enumerate(test_cases, start=1):
        request_id = f"test_{index}"
        request_output_dir = output_dir
        results_path = request_output_dir / "results" / f"{request_id}.json"
        workspace_dir = request_output_dir / "workspaces" / request_id
        status_path = request_output_dir / "status" / f"{request_id}.exit_code.txt"
        exit_code = status_path.read_text(encoding="utf-8").strip() if status_path.exists() else ""

        if not results_path.exists():
            error = "fast-agent run did not produce a results artifact."
            if exit_code and exit_code != "0":
                error = f"{error} Exit code: {exit_code}."
            results.append(TestResult(test_case=test_case, success=False, error=error))
            continue

        try:
            parsed = parse_fast_agent_results(results_path)
        except Exception as exc:
            results.append(TestResult(test_case=test_case, success=False, error=str(exc)))
            continue

        if exit_code and exit_code != "0":
            results.append(
                TestResult(
                    test_case=test_case,
                    success=False,
                    output=parsed.output_text,
                    error=f"fast-agent exited with code {exit_code}.",
                    stats=parsed.stats,
                    tokens_used=parsed.stats.total_tokens,
                    turns=parsed.stats.turns,
                )
            )
            continue

        success, validation_result = check_expected(
            parsed.output_text or "",
            test_case.expected,
            workspace_dir if workspace_dir.exists() else None,
            test_case,
        )
        results.append(
            TestResult(
                test_case=test_case,
                success=success,
                output=parsed.output_text,
                tokens_used=parsed.stats.total_tokens,
                turns=parsed.stats.turns,
                stats=parsed.stats,
                validation_result=validation_result,
            )
        )
    return results


def run_remote_eval(
    *,
    skill: Skill,
    test_cases: list[TestCase],
    model: str,
    jobs_config: JobsConfig,
    fastagent_config_path: Path,
    destination_root: Path,
    run_baseline: bool,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[EvalResults | None, list[str]]:
    """Run remote fast-agent batches and reconstruct ``EvalResults`` locally."""
    job_refs: list[str] = []
    submission_config = replace(jobs_config, wait=False)
    timeout_seconds = parse_duration_seconds(jobs_config.jobs_timeout)

    with_skill_submission = submit_fast_agent_eval_job(
        skill=skill,
        test_cases=test_cases,
        fastagent_config_path=fastagent_config_path,
        model=model,
        jobs_config=submission_config,
        destination_root=destination_root / "with-skill",
        phase_label="with-skill",
        progress_callback=progress_callback,
    )
    assert isinstance(with_skill_submission, SubmittedJob)
    job_refs.append(with_skill_submission.job_id)

    jobs_to_collect: list[tuple[str, SubmittedJob, Path]] = [
        ("with-skill", with_skill_submission, destination_root / "with-skill")
    ]

    if run_baseline:
        baseline_submission = submit_fast_agent_eval_job(
            skill=None,
            test_cases=test_cases,
            fastagent_config_path=fastagent_config_path,
            model=model,
            jobs_config=submission_config,
            destination_root=destination_root / "baseline",
            phase_label="baseline",
            progress_callback=progress_callback,
        )
        assert isinstance(baseline_submission, SubmittedJob)
        job_refs.append(baseline_submission.job_id)
        jobs_to_collect.append(("baseline", baseline_submission, destination_root / "baseline"))

    if not jobs_config.wait:
        return None, job_refs

    collected_outputs: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs_to_collect)) as executor:
        future_map = {
            executor.submit(
                wait_for_job_outputs,
                submission,
                destination_root=collect_root,
                wait_timeout_seconds=timeout_seconds,
                progress_callback=progress_callback,
            ): label
            for label, submission, collect_root in jobs_to_collect
        }
        for future in concurrent.futures.as_completed(future_map):
            label = future_map[future]
            collected_outputs[label] = future.result()

    results = EvalResults(skill_name=skill.name, model=model)
    results.with_skill_results = reconstruct_remote_eval_results(
        output_dir=collected_outputs["with-skill"],
        test_cases=test_cases,
    )

    if run_baseline:
        results.baseline_results = reconstruct_remote_eval_results(
            output_dir=collected_outputs["baseline"],
            test_cases=test_cases,
        )

    return apply_eval_metrics(results, test_cases), job_refs
