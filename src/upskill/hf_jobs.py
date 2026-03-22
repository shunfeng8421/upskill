"""Helpers for submitting and collecting Hugging Face Jobs-based eval runs."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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


_JOB_URL_RE = re.compile(r"https://huggingface\.co/jobs/(?P<namespace>[^/]+)/(?P<job_id>[^/\s]+)")
_HF_UPLOAD_CONFLICT_MARKERS = (
    "412 Precondition Failed",
    "A commit has happened since. Please refresh and try again.",
)
_HF_AUTH_RATE_LIMIT_MARKERS = (
    "rate limit for the /whoami-v2 endpoint",
    "whoami-v2",
)
_HF_SUBMISSION_LOCK = threading.RLock()
_VERIFIED_ARTIFACT_REPOS: set[str] = set()
_HF_JOBS_IMAGE = "ghcr.io/astral-sh/uv:python3.13-bookworm"
_HF_HUB_CLI_SPEC = "huggingface_hub[cli]==1.7.2"
_FAST_AGENT_SPEC = "fast-agent-mcp==0.6.2"


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


def _is_retryable_hf_upload_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    """Return whether a failed ``hf upload`` can be retried safely."""
    if completed.returncode == 0:
        return False
    output = f"{completed.stdout}\n{completed.stderr}"
    return any(marker in output for marker in _HF_UPLOAD_CONFLICT_MARKERS)


def _is_retryable_hf_auth_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    """Return whether a failed HF CLI call hit auth-related rate limiting."""
    if completed.returncode == 0:
        return False
    output = f"{completed.stdout}\n{completed.stderr}"
    return any(marker in output for marker in _HF_AUTH_RATE_LIMIT_MARKERS)


def _run_hf_command_with_retry(
    command: list[str],
    *,
    retryable: Callable[[subprocess.CompletedProcess[str]], bool],
    attempts: int = 5,
    initial_delay_seconds: float = 1.0,
) -> subprocess.CompletedProcess[str]:
    """Run an HF CLI command with retry/backoff for known transient failures."""
    delay_seconds = initial_delay_seconds
    last_completed: subprocess.CompletedProcess[str] | None = None

    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            command,
            cwd=_repo_root(),
            check=False,
            capture_output=True,
            text=True,
        )
        last_completed = completed
        if completed.returncode == 0:
            return completed
        if attempt >= attempts or not retryable(completed):
            return completed
        time.sleep(delay_seconds)
        delay_seconds *= 2

    if last_completed is None:
        raise RuntimeError("HF CLI retry loop completed without executing a command.")
    return last_completed


def _run_hf_upload_with_retry(
    command: list[str],
    *,
    attempts: int = 5,
    initial_delay_seconds: float = 1.0,
) -> subprocess.CompletedProcess[str]:
    """Run ``hf upload`` with retry/backoff for dataset branch conflicts."""
    return _run_hf_command_with_retry(
        command,
        retryable=_is_retryable_hf_upload_failure,
        attempts=attempts,
        initial_delay_seconds=initial_delay_seconds,
    )


def _run_hf_auth_command_with_retry(
    command: list[str],
    *,
    attempts: int = 5,
    initial_delay_seconds: float = 1.0,
) -> subprocess.CompletedProcess[str]:
    """Run HF CLI commands that may transiently fail on auth endpoint rate limits."""
    return _run_hf_command_with_retry(
        command,
        retryable=_is_retryable_hf_auth_failure,
        attempts=attempts,
        initial_delay_seconds=initial_delay_seconds,
    )


def _verify_artifact_repo_access(artifact_repo: str) -> None:
    """Verify that the configured artifact dataset repo exists and is accessible."""
    with _HF_SUBMISSION_LOCK:
        if artifact_repo in _VERIFIED_ARTIFACT_REPOS:
            return

        completed = _run_hf_auth_command_with_retry(
            [
                "hf",
                "download",
                artifact_repo,
                "--repo-type",
                "dataset",
                "--dry-run",
                "--quiet",
            ]
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Artifact repo is not accessible. Create it before submitting jobs and "
                "ensure the current Hugging Face credentials can access it:\n"
                f"repo: {artifact_repo}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        _VERIFIED_ARTIFACT_REPOS.add(artifact_repo)


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
    entropy = uuid.uuid4().hex[:8]
    prefix = f"{timestamp}_{suffix}" if suffix else timestamp
    return f"{prefix}_{entropy}"


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


def _submit_bundle_job(
    *,
    bundle_archive: Path,
    jobs_config: JobsConfig,
    run_id: str,
    model: str,
) -> SubmittedJob:
    _verify_artifact_repo_access(jobs_config.artifact_repo)
    with _HF_SUBMISSION_LOCK:
        upload = _run_hf_upload_with_retry(
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
            ]
        )
    if upload.returncode != 0:
        raise RuntimeError(
            "Failed to upload remote fast-agent bundle:\n"
            f"stdout:\n{upload.stdout}\n"
            f"stderr:\n{upload.stderr}"
        )

    job_cmd = (
        "set -euo pipefail\n"
        "upload_with_retries() {\n"
        '  local repo="$1"\n'
        '  local src="$2"\n'
        '  local dest="$3"\n'
        '  local message="$4"\n'
        "  local delay=1\n"
        "  local attempt\n"
        "  for attempt in 1 2 3 4 5; do\n"
        '    local log_file="$(mktemp)"\n'
        '    if hf upload "$repo" "$src" "$dest" --repo-type dataset --commit-message "$message" >"$log_file" 2>&1; then\n'
        '      cat "$log_file"\n'
        '      rm -f "$log_file"\n'
        "      return 0\n"
        "    fi\n"
        '    if grep -q "412 Precondition Failed" "$log_file" && [[ "$attempt" -lt 5 ]]; then\n'
        '      cat "$log_file" >&2\n'
        '      rm -f "$log_file"\n'
        '      sleep "$delay"\n'
        "      delay=$((delay * 2))\n"
        "      continue\n"
        "    fi\n"
        '    if grep -q "A commit has happened since" "$log_file" && [[ "$attempt" -lt 5 ]]; then\n'
        '      cat "$log_file" >&2\n'
        '      rm -f "$log_file"\n'
        '      sleep "$delay"\n'
        "      delay=$((delay * 2))\n"
        "      continue\n"
        "    fi\n"
        '    cat "$log_file" >&2\n'
        '    rm -f "$log_file"\n'
        "    return 1\n"
        "  done\n"
        "  return 1\n"
        "}\n"
        "WORK=/workspace\n"
        'mkdir -p "$WORK/out"\n'
        'cd "$WORK"\n'
        f'uv pip install --system "{_HF_HUB_CLI_SPEC}" "{_FAST_AGENT_SPEC}"\n'
        'hf download "$ARTIFACT_REPO" "inputs/$RUN_ID/bundle.tar.gz" --repo-type dataset --local-dir "$WORK"\n'
        'tar -xzf "$WORK/inputs/$RUN_ID/bundle.tar.gz" -C "$WORK"\n'
        "set +e\n"
        'bash "$WORK/bundle/job_entrypoint.sh" "$WORK/bundle" "$WORK/out"\n'
        "status=$?\n"
        "set -e\n"
        'echo "$status" > "$WORK/out/exit_code.txt"\n'
        'upload_with_retries "$ARTIFACT_REPO" "$WORK/out" "outputs/$RUN_ID" '
        '"outputs: $RUN_ID (exit=$status)"\n'
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
            _HF_JOBS_IMAGE,
            "bash",
            "-lc",
            job_cmd,
        ]
    )
    with _HF_SUBMISSION_LOCK:
        completed = _run_hf_auth_command_with_retry(command)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to submit remote fast-agent job:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    job_ref = _normalize_job_id(completed.stdout.strip().splitlines()[-1])
    return SubmittedJob(job_id=job_ref, run_id=run_id, artifact_repo=jobs_config.artifact_repo)
