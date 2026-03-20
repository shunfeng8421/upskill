"""Helpers for building fast-agent CLI invocations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from upskill.executors.contracts import ExecutionRequest


def build_fast_agent_command(
    request: ExecutionRequest,
    *,
    skills_dir: Path,
    results_path: Path,
    fast_agent_bin: str = "fast-agent",
) -> list[str]:
    """Build the canonical fast-agent automation command for a request."""
    command = [
        fast_agent_bin,
        "go",
        "--model",
        request.model,
        "--skills-dir",
        str(skills_dir),
        "--results",
        str(results_path),
        "--quiet",
    ]
    if request.enable_shell:
        command.append("--shell")
    command.extend(["--message", request.prompt])
    return command
