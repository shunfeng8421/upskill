"""Remote HF Jobs-backed executor for fast-agent evaluation."""

from __future__ import annotations

import asyncio
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from upskill.artifacts import (
    bundle_agent_card,
    copy_config_file,
    ensure_directory,
    materialize_skill_bundle,
    materialize_workspace,
    write_request_file,
)
from upskill.executors.contracts import ExecutionHandle, ExecutionRequest, ExecutionResult
from upskill.hf_jobs import (
    JobsConfig,
    SubmittedJob,
    _make_run_id,
    _sanitize_hf_job_label_value,
    _submit_bundle_job,
    parse_duration_seconds,
    wait_for_job_outputs,
)
from upskill.models import ConversationStats
from upskill.result_parsing import parse_fast_agent_results
from upskill.trace_export import build_trace_dataset_path

if TYPE_CHECKING:
    from collections.abc import Callable


class RemoteFastAgentExecutor:
    """Execute evaluation requests by submitting one HF job per request."""

    def __init__(
        self,
        *,
        jobs_config: JobsConfig,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._jobs_config = jobs_config
        self._progress_callback = progress_callback

    async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
        """Start a remote job-backed execution."""
        task = asyncio.create_task(asyncio.to_thread(self._run_request_sync, request))
        return ExecutionHandle(request=request, task=task)

    async def submit(self, request: ExecutionRequest) -> SubmittedJob:
        """Submit a remote execution request without waiting for results."""
        return await asyncio.to_thread(self._submit_request_sync, request)

    async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        """Collect a previously started remote execution."""
        return await handle.task

    async def cancel(self, handle: ExecutionHandle) -> None:
        """Cancel a previously started remote execution."""
        handle.task.cancel()
        try:
            await handle.task
        except asyncio.CancelledError:
            return

    def _submit_request_sync(self, request: ExecutionRequest) -> SubmittedJob:
        normalized_request, artifact_dir = self._prepare_request(request)
        return self._submit_prepared_request(normalized_request, artifact_dir)

    def _submit_prepared_request(
        self,
        request: ExecutionRequest,
        artifact_dir: Path,
    ) -> SubmittedJob:
        temp_root, bundle_archive = self._create_bundle_archive(request)
        try:
            submission = self._submit_bundle(request, bundle_archive)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        request_path = artifact_dir / "submitted_job.json"
        request_path.write_text(
            json.dumps(
                {
                    "job_id": submission.job_id,
                    "run_id": submission.run_id,
                    "artifact_repo": submission.artifact_repo,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return submission

    def _run_request_sync(self, request: ExecutionRequest) -> ExecutionResult:
        normalized_request, artifact_dir = self._prepare_request(request)
        workspace_dir = artifact_dir / "workspace"
        submission = self._submit_prepared_request(normalized_request, artifact_dir)

        remote_output_dir = wait_for_job_outputs(
            submission,
            destination_root=artifact_dir / "remote_download",
            wait_timeout_seconds=parse_duration_seconds(self._jobs_config.jobs_timeout),
            progress_callback=self._progress_callback,
        )

        stdout_path = artifact_dir / "stdout.txt"
        stderr_path = artifact_dir / "stderr.txt"
        results_path = artifact_dir / "results.json"
        self._materialize_remote_outputs(
            remote_output_dir=remote_output_dir,
            artifact_dir=artifact_dir,
            workspace_dir=workspace_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            results_path=results_path,
        )

        exit_code = self._read_exit_code(remote_output_dir)
        error: str | None = None
        parsed_output: str | None = None
        parsed_stats = ConversationStats()

        if not results_path.exists():
            error = "fast-agent run did not produce a results artifact."
        else:
            try:
                parsed = parse_fast_agent_results(results_path)
            except Exception as exc:
                error = f"Failed to parse fast-agent results: {exc}"
            else:
                parsed_output = parsed.output_text
                parsed_stats = parsed.stats

        if exit_code not in {"", "0"}:
            exit_error = f"fast-agent exited with code {exit_code}."
            error = f"{error} {exit_error}".strip() if error else exit_error

        metadata = {
            **normalized_request.metadata,
            "job_id": submission.job_id,
            "run_id": submission.run_id,
            "return_code": int(exit_code) if exit_code else None,
        }
        return ExecutionResult(
            output_text=parsed_output,
            raw_results_path=results_path if results_path.exists() else None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifact_dir=artifact_dir,
            workspace_dir=workspace_dir,
            stats=parsed_stats,
            error=error,
            metadata=metadata,
        )

    def _prepare_request(self, request: ExecutionRequest) -> tuple[ExecutionRequest, Path]:
        artifact_dir = ensure_directory(request.artifact_dir.resolve())
        normalized_request = ExecutionRequest(
            prompt=request.prompt,
            model=request.model,
            agent=request.agent,
            fastagent_config_path=request.fastagent_config_path.resolve(),
            artifact_dir=artifact_dir,
            cards_source_dir=request.cards_source_dir.resolve(),
            label=request.label,
            skill=request.skill,
            workspace_files=dict(request.workspace_files),
            metadata=dict(request.metadata),
        )
        workspace_dir = ensure_directory(artifact_dir / "workspace")
        materialize_workspace(workspace_dir, normalized_request.workspace_files)

        bundle_agent_card(
            normalized_request.cards_source_dir,
            artifact_dir / "cards",
            agent_name=normalized_request.agent,
        )
        materialize_skill_bundle(artifact_dir / "skills", normalized_request)
        copy_config_file(
            normalized_request.fastagent_config_path,
            artifact_dir / "fastagent.config.yaml",
        )
        copy_config_file(
            normalized_request.fastagent_config_path,
            workspace_dir / "fastagent.config.yaml",
        )

        request_path = artifact_dir / "request.json"
        write_request_file(request_path, normalized_request)
        (artifact_dir / "prompt.txt").write_text(normalized_request.prompt, encoding="utf-8")
        return normalized_request, artifact_dir

    def _submit_bundle(self, request: ExecutionRequest, bundle_archive: Path) -> SubmittedJob:
        run_id = _make_run_id("request", request.model, request.label)
        labels = self._build_job_labels(request, run_id=run_id)
        trace_dataset_path = build_trace_dataset_path(request, run_id=run_id)
        submission = _submit_bundle_job(
            bundle_archive=bundle_archive,
            jobs_config=self._jobs_config,
            run_id=run_id,
            model=request.model,
            labels=labels,
            trace_dataset_path=trace_dataset_path,
            trace_agent=request.agent,
        )
        if self._progress_callback is not None:
            self._progress_callback(
                f"submitted remote request {request.label} as job "
                f"{submission.job_id} (run_id={submission.run_id})"
            )
        return submission

    def _build_job_labels(self, request: ExecutionRequest, *, run_id: str) -> dict[str, str]:
        operation = request.metadata.get("operation")
        labels = {
            "upskill-agent": _sanitize_hf_job_label_value(request.agent, default="agent"),
            "upskill-executor": "remote-fast-agent",
            "upskill-model": _sanitize_hf_job_label_value(request.model, default="model"),
            "upskill-operation": _sanitize_hf_job_label_value(
                operation if isinstance(operation, str) else "eval",
                default="eval",
            ),
            "upskill-request": _sanitize_hf_job_label_value(request.label, default="request"),
            "upskill-run-id": _sanitize_hf_job_label_value(run_id, default="run"),
        }
        if request.skill is not None:
            labels["upskill-skill"] = _sanitize_hf_job_label_value(
                request.skill.name,
                default="skill",
            )
        return labels

    def _create_bundle_archive(self, request: ExecutionRequest) -> tuple[Path, Path]:
        temp_root = Path(tempfile.mkdtemp(prefix="upskill_hf_request_"))
        bundle_root = temp_root / "bundle"
        ensure_directory(bundle_root)
        ensure_directory(bundle_root / "skills")
        if request.skill is not None:
            request.skill.save(bundle_root / "skills" / request.skill.name)
        bundle_agent_card(
            request.cards_source_dir,
            bundle_root / "cards",
            agent_name=request.agent,
        )
        copy_config_file(request.fastagent_config_path, bundle_root / "fastagent.config.yaml")
        (bundle_root / "agent.txt").write_text(request.agent, encoding="utf-8")

        request_dir = ensure_directory(bundle_root / "requests" / "request_1")
        (request_dir / "prompt.txt").write_text(request.prompt, encoding="utf-8")
        request_workspace_dir = ensure_directory(request_dir / "workspace")
        materialize_workspace(request_workspace_dir, request.workspace_files)
        (bundle_root / "manifest.json").write_text(
            json.dumps(
                {
                    "request_count": 1,
                    "requests": [
                        {
                            "id": "request_1",
                            "index": 1,
                            "has_workspace_files": bool(request.workspace_files),
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        entrypoint_source = (
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "hf"
            / "job_entrypoint_eval_fast_agent.sh"
        )
        shutil.copy2(entrypoint_source, bundle_root / "job_entrypoint.sh")
        bundle_archive = temp_root / "bundle.tar.gz"
        with tarfile.open(bundle_archive, "w:gz") as archive:
            archive.add(bundle_root, arcname="bundle")
        return temp_root, bundle_archive

    def _materialize_remote_outputs(
        self,
        *,
        remote_output_dir: Path,
        artifact_dir: Path,
        workspace_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
        results_path: Path,
    ) -> None:
        preserved_output_dir = artifact_dir / "remote_output"
        if preserved_output_dir.exists():
            shutil.rmtree(preserved_output_dir)
        shutil.copytree(remote_output_dir, preserved_output_dir)

        remote_stdout = remote_output_dir / "logs" / "request_1.out.txt"
        remote_stderr = remote_output_dir / "logs" / "request_1.err.txt"
        remote_results = remote_output_dir / "results" / "request_1.json"
        remote_workspace = remote_output_dir / "workspaces" / "request_1"

        if remote_stdout.exists():
            shutil.copy2(remote_stdout, stdout_path)
        else:
            stdout_path.write_text("", encoding="utf-8")
        if remote_stderr.exists():
            shutil.copy2(remote_stderr, stderr_path)
        else:
            stderr_path.write_text("", encoding="utf-8")
        if remote_results.exists():
            shutil.copy2(remote_results, results_path)
        if remote_workspace.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
            shutil.copytree(remote_workspace, workspace_dir)

    def _read_exit_code(self, remote_output_dir: Path) -> str:
        status_path = remote_output_dir / "status" / "request_1.exit_code.txt"
        if not status_path.exists():
            return ""
        return status_path.read_text(encoding="utf-8").strip()
