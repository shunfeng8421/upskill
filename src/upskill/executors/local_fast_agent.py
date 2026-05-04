"""Local shell-out executor for fast-agent-backed evaluation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
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
from upskill.fast_agent_cli import build_fast_agent_command
from upskill.models import ConversationStats
from upskill.result_parsing import parse_fast_agent_results
from upskill.trace_export import build_trace_dataset_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class LocalFastAgentExecutor:
    """Execute evaluation requests by shelling out to ``fast-agent`` locally."""

    def __init__(
        self,
        *,
        fast_agent_bin: str = "fast-agent",
        artifact_repo: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._fast_agent_bin = fast_agent_bin
        self._artifact_repo = artifact_repo
        self._progress_callback = progress_callback

    async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
        """Start a local subprocess execution."""
        task = asyncio.create_task(self._run_request(request))
        return ExecutionHandle(request=request, task=task)

    async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        """Collect a previously started subprocess execution."""
        return await handle.task

    async def cancel(self, handle: ExecutionHandle) -> None:
        """Cancel a previously started subprocess execution."""
        handle.task.cancel()
        try:
            await handle.task
        except asyncio.CancelledError:
            return

    async def _run_request(self, request: ExecutionRequest) -> ExecutionResult:
        artifact_dir = ensure_directory(request.artifact_dir.resolve())
        normalized_request = replace(
            request,
            fastagent_config_path=request.fastagent_config_path.resolve(),
            artifact_dir=artifact_dir,
            cards_source_dir=request.cards_source_dir.resolve(),
        )
        workspace_dir = ensure_directory(artifact_dir / "workspace")
        materialize_workspace(workspace_dir, normalized_request.workspace_files)

        cards_dir = bundle_agent_card(
            normalized_request.cards_source_dir,
            artifact_dir / "cards",
            agent_name=normalized_request.agent,
        )
        skills_dir = materialize_skill_bundle(artifact_dir / "skills", normalized_request)
        preserved_config_path = copy_config_file(
            normalized_request.fastagent_config_path,
            artifact_dir / "fastagent.config.yaml",
        )
        workspace_config_path = copy_config_file(
            normalized_request.fastagent_config_path,
            workspace_dir / "fastagent.config.yaml",
        )
        del preserved_config_path, workspace_config_path

        request_path = artifact_dir / "request.json"
        write_request_file(request_path, normalized_request)

        prompt_path = artifact_dir / "prompt.txt"
        prompt_path.write_text(normalized_request.prompt, encoding="utf-8")

        results_path = artifact_dir / "results.json"
        stdout_path = artifact_dir / "stdout.txt"
        stderr_path = artifact_dir / "stderr.txt"
        command = build_fast_agent_command(
            normalized_request,
            config_path=normalized_request.fastagent_config_path
            if normalized_request.fastagent_config_path.exists()
            else None,
            cards_dir=cards_dir,
            skills_dir=skills_dir,
            prompt_path=prompt_path,
            results_path=results_path,
            fast_agent_bin=self._fast_agent_bin,
        )
        command_path = artifact_dir / "command.json"
        command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace_dir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")

        error: str | None = None
        parsed_output: str | None = None
        parsed_stats = None

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

        if process.returncode != 0:
            exit_error = f"fast-agent exited with code {process.returncode}."
            error = f"{error} {exit_error}".strip() if error else exit_error

        metadata = {
            **normalized_request.metadata,
            "return_code": process.returncode,
        }
        trace_error = await self._export_trace(normalized_request, artifact_dir)
        if trace_error is not None:
            metadata["trace_export_error"] = trace_error

        result = ExecutionResult(
            output_text=parsed_output,
            raw_results_path=results_path if results_path.exists() else None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifact_dir=artifact_dir,
            workspace_dir=workspace_dir,
            stats=parsed_stats or ConversationStats(),
            error=error,
            metadata=metadata,
        )
        return result

    async def _export_trace(self, request: ExecutionRequest, artifact_dir: Path) -> str | None:
        """Export and publish the latest fast-agent trace when an artifact repo is configured."""
        if self._artifact_repo is None:
            return None

        trace_path = artifact_dir / "trace.jsonl"
        dataset_path = build_trace_dataset_path(request)
        if self._progress_callback is not None:
            self._progress_callback(f"exporting trace for {request.label} to {dataset_path}")
        export_stdout_path = artifact_dir / "trace_export_stdout.txt"
        export_stderr_path = artifact_dir / "trace_export_stderr.txt"
        command = [
            self._fast_agent_bin,
            "export",
            "latest",
            "--agent",
            request.agent,
            "--output",
            str(trace_path),
            "--hf-dataset",
            self._artifact_repo,
            "--hf-dataset-path",
            dataset_path,
        ]
        (artifact_dir / "trace_export_command.json").write_text(
            json.dumps(command, indent=2),
            encoding="utf-8",
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=artifact_dir,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        export_stdout_path.write_text(
            stdout_bytes.decode("utf-8", errors="replace"), encoding="utf-8"
        )
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")
        export_stderr_path.write_text(stderr_text, encoding="utf-8")

        if process.returncode != 0:
            return f"fast-agent trace export exited with code {process.returncode}."
        if self._progress_callback is not None:
            self._progress_callback(f"exported trace for {request.label} to {dataset_path}")
        return None
