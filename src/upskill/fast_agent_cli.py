"""Helpers for building fast-agent CLI invocations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from upskill.executors.contracts import ExecutionRequest


def build_fast_agent_command(
    request: ExecutionRequest,
    *,
    config_path: Path | None,
    cards_dir: Path,
    skills_dir: Path,
    prompt_path: Path,
    results_path: Path,
    fast_agent_bin: str = "fast-agent",
) -> list[str]:
    """Build the canonical fast-agent automation command for a request."""
    command = [fast_agent_bin, "go"]
    if config_path is not None:
        command.extend(["--config-path", str(config_path)])
    command.extend(
        [
            "--card",
            str(cards_dir),
            "--agent",
            request.agent,
            "--model",
            request.model,
            "--skills-dir",
            str(skills_dir),
            "--prompt-file",
            str(prompt_path),
            "--results",
            str(results_path),
            "--quiet",
        ]
    )
    return command
