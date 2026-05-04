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
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class JobsConfig:
    """Configuration for remote Jobs-backed execution."""

    artifact_repo: str
    wait: bool = True
    jobs_timeout: str = "2h"
    jobs_flavor: str = "cpu-basic"
    jobs_secrets: str = "HF_TOKEN"
    jobs_namespace: str | None = None
    jobs_image: str = "ghcr.io/astral-sh/uv:python3.13-bookworm"


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
_FAST_AGENT_SPEC = "fast-agent-mcp==0.6.26"
_MAX_HF_JOB_LABEL_VALUE_LENGTH = 63
_HF_RETRY_ATTEMPTS = 5
_HF_INITIAL_RETRY_DELAY_SECONDS = 2.0


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


def _namespace_from_repo_id(repo_id: str) -> str | None:
    if "/" not in repo_id:
        return None
    namespace, _repo_name = repo_id.split("/", 1)
    normalized = namespace.strip()
    return normalized or None


def _resolve_jobs_namespace(
    *,
    job_id: str | None = None,
    artifact_repo: str | None = None,
    configured_namespace: str | None = None,
) -> str | None:
    if configured_namespace:
        return configured_namespace
    if job_id is not None:
        namespace, _bare_job_id = _split_job_reference(job_id)
        if namespace is not None:
            return namespace
    if artifact_repo is not None:
        return _namespace_from_repo_id(artifact_repo)
    return None


def _lookup_job_stage(job_id: str, *, namespace: str | None = None) -> str | None:
    """Best-effort lookup of an HF job stage from ``hf jobs ps --format json``."""
    bare_namespace, bare_job_id = _split_job_reference(job_id)
    resolved_namespace = _resolve_jobs_namespace(
        job_id=job_id,
        configured_namespace=namespace or bare_namespace,
    )
    command = ["hf", "jobs", "ps", "-a", "--format", "json"]
    if resolved_namespace is not None:
        command.extend(["--namespace", resolved_namespace])
    completed = _run_hf_command(command)
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("id", "")) != bare_job_id:
            continue
        owner = entry.get("owner")
        owner_name = owner.get("name") if isinstance(owner, dict) else None
        if resolved_namespace is not None and owner_name != resolved_namespace:
            continue
        status = entry.get("status")
        if isinstance(status, dict):
            stage = status.get("stage")
            if isinstance(stage, str):
                return stage
    return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hf_command_output(completed: subprocess.CompletedProcess[str]) -> str:
    """Return combined stdout/stderr for retry classification."""
    return f"{completed.stdout}\n{completed.stderr}"


def _has_retryable_hf_failure(
    completed: subprocess.CompletedProcess[str],
    *,
    markers: tuple[str, ...],
) -> bool:
    """Return whether a failed HF CLI call matches any retryable marker."""
    if completed.returncode == 0:
        return False
    output = _hf_command_output(completed)
    return any(marker in output for marker in markers)


def _is_retryable_hf_upload_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    """Return whether a failed ``hf upload`` can be retried safely."""
    return _has_retryable_hf_failure(completed, markers=_HF_UPLOAD_CONFLICT_MARKERS)


def _is_retryable_hf_auth_failure(completed: subprocess.CompletedProcess[str]) -> bool:
    """Return whether a failed HF CLI call hit auth-related rate limiting."""
    return _has_retryable_hf_failure(completed, markers=_HF_AUTH_RATE_LIMIT_MARKERS)


def _is_retryable_hf_failure(
    completed: subprocess.CompletedProcess[str],
    *,
    retry_auth_rate_limit: bool,
    retry_upload_conflicts: bool,
) -> bool:
    """Return whether a failed HF CLI call should be retried."""
    auth_retry = retry_auth_rate_limit and _is_retryable_hf_auth_failure(completed)
    upload_retry = retry_upload_conflicts and _is_retryable_hf_upload_failure(completed)
    return auth_retry or upload_retry


def _retry_exhausted_hf_failure_message(completed: subprocess.CompletedProcess[str]) -> str:
    """Return extra context when a retryable HF failure still exhausted retries."""
    if _is_retryable_hf_auth_failure(completed):
        return (
            "The Hugging Face CLI continued hitting the /whoami-v2 auth rate limit after "
            "retrying.\n"
        )
    if _is_retryable_hf_upload_failure(completed):
        return "The Hugging Face CLI continued hitting a retryable upload conflict.\n"
    return ""


def _run_hf_command_with_retry(
    command: list[str],
    *,
    retryable: Callable[[subprocess.CompletedProcess[str]], bool],
    attempts: int = _HF_RETRY_ATTEMPTS,
    initial_delay_seconds: float = _HF_INITIAL_RETRY_DELAY_SECONDS,
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


def _run_hf_command(
    command: list[str],
    *,
    retry_auth_rate_limit: bool = True,
    retry_upload_conflicts: bool = False,
    attempts: int = _HF_RETRY_ATTEMPTS,
    initial_delay_seconds: float = _HF_INITIAL_RETRY_DELAY_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run an HF CLI command through the shared retry policy."""
    return _run_hf_command_with_retry(
        command,
        retryable=lambda completed: _is_retryable_hf_failure(
            completed,
            retry_auth_rate_limit=retry_auth_rate_limit,
            retry_upload_conflicts=retry_upload_conflicts,
        ),
        attempts=attempts,
        initial_delay_seconds=initial_delay_seconds,
    )


def _verify_artifact_repo_access(artifact_repo: str) -> None:
    """Verify that the configured artifact dataset repo exists and is accessible."""
    with _HF_SUBMISSION_LOCK:
        if artifact_repo in _VERIFIED_ARTIFACT_REPOS:
            return

        completed = _run_hf_command(
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
                f"{_retry_exhausted_hf_failure_message(completed)}"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        _VERIFIED_ARTIFACT_REPOS.add(artifact_repo)


def verify_artifact_repo_access(artifact_repo: str) -> None:
    """Validate that the artifact dataset repo exists and is accessible."""
    _verify_artifact_repo_access(artifact_repo)


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


def _sanitize_hf_job_label_value(value: str, *, default: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    truncated = sanitized[:_MAX_HF_JOB_LABEL_VALUE_LENGTH].strip("-")
    return truncated or default


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
        stage = _lookup_job_stage(
            job.job_id,
            namespace=_resolve_jobs_namespace(
                job_id=job.job_id,
                artifact_repo=job.artifact_repo,
            ),
        )
        marker_download = _run_hf_command(
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
        )
        if marker_download.returncode == 0:
            if progress_callback is not None:
                progress_callback(f"job {job.job_id} completed; downloading artifacts")
            full_download = _run_hf_command(
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
            )
            if full_download.returncode != 0:
                raise RuntimeError(
                    "HF job finished but artifacts could not be downloaded:\n"
                    f"{_retry_exhausted_hf_failure_message(full_download)}"
                    f"stdout:\n{full_download.stdout}\n"
                    f"stderr:\n{full_download.stderr}"
                )
            if progress_callback is not None:
                progress_callback(f"downloaded artifacts for job {job.job_id}")
            return destination_root / "outputs" / job.run_id
        if _is_retryable_hf_auth_failure(marker_download):
            raise RuntimeError(
                "Failed to check remote fast-agent job outputs after repeated Hugging Face "
                "auth retries:\n"
                f"stdout:\n{marker_download.stdout}\n"
                f"stderr:\n{marker_download.stderr}"
            )
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


def _hf_label_flags(labels: Mapping[str, str] | None) -> list[str]:
    flags: list[str] = []
    if not labels:
        return flags
    for key, value in sorted(labels.items()):
        flags.extend(["--label", f"{key}={value}"])
    return flags


def _upload_bundle_input(
    *,
    bundle_archive: Path,
    artifact_repo: str,
    run_id: str,
) -> subprocess.CompletedProcess[str]:
    """Upload a prepared request bundle into the artifact dataset."""
    with _HF_SUBMISSION_LOCK:
        return _run_hf_command(
            [
                "hf",
                "upload",
                artifact_repo,
                str(bundle_archive),
                f"inputs/{run_id}/bundle.tar.gz",
                "--repo-type",
                "dataset",
                "--commit-message",
                f"inputs: {run_id}",
            ],
            retry_upload_conflicts=True,
        )


def _render_bundle_job_script() -> str:
    """Render the shell script executed inside the remote HF job container."""
    return "\n".join(
        [
            "set -euo pipefail",
            "run_hf_with_retries() {",
            "  local delay=2",
            "  local attempt",
            f"  for attempt in $(seq 1 {_HF_RETRY_ATTEMPTS}); do",
            '    local log_file="$(mktemp)"',
            '    if "$@" >"$log_file" 2>&1; then',
            '      cat "$log_file"',
            '      rm -f "$log_file"',
            "      return 0",
            "    fi",
            f'    if [[ "$attempt" -lt {_HF_RETRY_ATTEMPTS} ]] && (',
            '      grep -q "rate limit for the /whoami-v2 endpoint" "$log_file" ||',
            '      grep -q "whoami-v2" "$log_file" ||',
            '      grep -q "412 Precondition Failed" "$log_file" ||',
            '      grep -q "A commit has happened since" "$log_file"',
            "    ); then",
            '      cat "$log_file" >&2',
            '      rm -f "$log_file"',
            '      sleep "$delay"',
            "      delay=$((delay * 2))",
            "      continue",
            "    fi",
            '    cat "$log_file" >&2',
            '    rm -f "$log_file"',
            "    return 1",
            "  done",
            "  return 1",
            "}",
            "download_with_retries() {",
            '  local repo="$1"',
            '  local path="$2"',
            '  local local_dir="$3"',
            '  run_hf_with_retries hf download "$repo" "$path" --repo-type dataset --local-dir "$local_dir"',
            "}",
            "upload_with_retries() {",
            '  local repo="$1"',
            '  local src="$2"',
            '  local dest="$3"',
            '  local message="$4"',
            '  run_hf_with_retries hf upload "$repo" "$src" "$dest" --repo-type dataset --commit-message "$message"',
            "}",
            "WORK=/workspace",
            'mkdir -p "$WORK/out"',
            'cd "$WORK"',
            'export FAST_AGENT_ENV_DIR="$WORK/out/fast-agent-env"',
            f'uv pip install --system "{_FAST_AGENT_SPEC}"',
            'download_with_retries "$ARTIFACT_REPO" "inputs/$RUN_ID/bundle.tar.gz" "$WORK"',
            'tar -xzf "$WORK/inputs/$RUN_ID/bundle.tar.gz" -C "$WORK"',
            "set +e",
            'bash "$WORK/bundle/job_entrypoint.sh" "$WORK/bundle" "$WORK/out"',
            "status=$?",
            "set -e",
            'echo "$status" > "$WORK/out/exit_code.txt"',
            "set +e",
            'fast-agent --env "$FAST_AGENT_ENV_DIR" export latest --agent "$FAST_AGENT_EXPORT_AGENT" '
            '--output "$WORK/out/trace.jsonl" --hf-dataset "$ARTIFACT_REPO" '
            '--hf-dataset-path "$TRACE_DATASET_PATH" >"$WORK/out/trace_export_stdout.txt" '
            '2>"$WORK/out/trace_export_stderr.txt"',
            'echo "$?" > "$WORK/out/trace_export_exit_code.txt"',
            "set -e",
            'upload_with_retries "$ARTIFACT_REPO" "$WORK/out" "outputs/$RUN_ID" '
            '"outputs: $RUN_ID (exit=$status)"',
            'exit "$status"',
            "",
        ]
    )


def _build_hf_jobs_run_command(
    *,
    jobs_config: JobsConfig,
    run_id: str,
    model: str,
    labels: Mapping[str, str] | None,
    job_script: str,
    trace_dataset_path: str,
    trace_agent: str,
) -> list[str]:
    """Build the ``hf jobs run`` command for a prepared bundle submission."""
    namespace = _resolve_jobs_namespace(
        artifact_repo=jobs_config.artifact_repo,
        configured_namespace=jobs_config.jobs_namespace,
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
        *_hf_label_flags(labels),
        "--env",
        f"ARTIFACT_REPO={jobs_config.artifact_repo}",
        "--env",
        f"RUN_ID={run_id}",
        "--env",
        f"FAST_MODEL={model}",
        "--env",
        f"FAST_AGENT_EXPORT_AGENT={trace_agent}",
        "--env",
        f"TRACE_DATASET_PATH={trace_dataset_path}",
    ]
    if namespace is not None:
        command.extend(["--namespace", namespace])
    command.extend(
        [
            "--",
            jobs_config.jobs_image,
            "bash",
            "-lc",
            job_script,
        ]
    )
    return command


def _submit_prepared_bundle_job(
    *,
    jobs_config: JobsConfig,
    run_id: str,
    model: str,
    labels: Mapping[str, str] | None = None,
    trace_dataset_path: str | None = None,
    trace_agent: str = "evaluator",
) -> SubmittedJob:
    """Submit a remote job for a bundle that is already present in the dataset."""
    job_script = _render_bundle_job_script()
    command = _build_hf_jobs_run_command(
        jobs_config=jobs_config,
        run_id=run_id,
        model=model,
        labels=labels,
        job_script=job_script,
        trace_dataset_path=trace_dataset_path or f"traces/eval/{model}/{run_id}.json",
        trace_agent=trace_agent,
    )
    with _HF_SUBMISSION_LOCK:
        completed = _run_hf_command(command)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to submit remote fast-agent job:\n"
            f"{_retry_exhausted_hf_failure_message(completed)}"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    job_ref = _normalize_job_id(completed.stdout.strip().splitlines()[-1])
    return SubmittedJob(job_id=job_ref, run_id=run_id, artifact_repo=jobs_config.artifact_repo)


def _submit_bundle_job(
    *,
    bundle_archive: Path,
    jobs_config: JobsConfig,
    run_id: str,
    model: str,
    labels: Mapping[str, str] | None = None,
    trace_dataset_path: str | None = None,
    trace_agent: str = "evaluator",
) -> SubmittedJob:
    upload = _upload_bundle_input(
        bundle_archive=bundle_archive,
        artifact_repo=jobs_config.artifact_repo,
        run_id=run_id,
    )
    if upload.returncode != 0:
        raise RuntimeError(
            "Failed to upload remote fast-agent bundle:\n"
            f"{_retry_exhausted_hf_failure_message(upload)}"
            f"stdout:\n{upload.stdout}\n"
            f"stderr:\n{upload.stderr}"
        )
    return _submit_prepared_bundle_job(
        jobs_config=jobs_config,
        run_id=run_id,
        model=model,
        labels=labels,
        trace_dataset_path=trace_dataset_path,
        trace_agent=trace_agent,
    )
