"""Local shell-out executor for fast-agent-backed evaluation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from upskill.artifacts import (
    bundle_cards,
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


class LocalFastAgentExecutor:
    """Execute evaluation requests by shelling out to ``fast-agent`` locally."""

    def __init__(self, *, fast_agent_bin: str = "fast-agent") -> None:
        self._fast_agent_bin = fast_agent_bin

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

        cards_dir = bundle_cards(normalized_request.cards_source_dir, artifact_dir / "cards")
        skills_dir = materialize_skill_bundle(artifact_dir / "skills", normalized_request)
        preserved_config_path = copy_config_file(
            normalized_request.fastagent_config_path,
            artifact_dir / "fastagent.config.yaml",
        )
        del preserved_config_path

        request_path = artifact_dir / "request.json"
        write_request_file(request_path, normalized_request)

        prompt_path = artifact_dir / "prompt.txt"
        prompt_path.write_text(normalized_request.prompt, encoding="utf-8")

        results_path = artifact_dir / "results.json"
        stdout_path = artifact_dir / "stdout.txt"
        stderr_path = artifact_dir / "stderr.txt"
        command = build_fast_agent_command(
            normalized_request,
            cards_dir=cards_dir,
            skills_dir=skills_dir,
            results_path=results_path,
            prompt_file=prompt_path,
            fast_agent_bin=self._fast_agent_bin,
        )
        command_path = artifact_dir / "command.json"
        command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=workspace_dir,
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

        result = ExecutionResult(
            output_text=parsed_output,
            raw_results_path=results_path if results_path.exists() else None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifact_dir=artifact_dir,
            workspace_dir=workspace_dir,
            stats=parsed_stats or ConversationStats(),
            error=error,
            metadata={
                **normalized_request.metadata,
                "return_code": process.returncode,
            },
        )
        return result
