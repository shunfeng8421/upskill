"""Execution request and result contracts for evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from upskill.models import ConversationStats, Skill

if TYPE_CHECKING:
    import asyncio
    from pathlib import Path

ExecutionMetadataValue = str | int | float | bool | None


@dataclass(slots=True)
class ExecutionRequest:
    """Semantic execution request for a single evaluation run."""

    prompt: str
    model: str
    agent: str
    fastagent_config_path: Path
    artifact_dir: Path
    cards_source_dir: Path
    label: str
    skill: Skill | None = None
    workspace_files: dict[str, str] = field(default_factory=dict)
    enable_shell: bool = False
    metadata: dict[str, ExecutionMetadataValue] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionResult:
    """Collected execution result plus preserved artifact paths."""

    output_text: str | None
    raw_results_path: Path | None
    stdout_path: Path
    stderr_path: Path
    artifact_dir: Path
    workspace_dir: Path
    stats: ConversationStats = field(default_factory=ConversationStats)
    error: str | None = None
    metadata: dict[str, ExecutionMetadataValue] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionHandle:
    """In-flight execution handle."""

    request: ExecutionRequest
    task: asyncio.Task[ExecutionResult]
