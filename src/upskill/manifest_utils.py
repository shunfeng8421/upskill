"""Skill manifest utilities for upskill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from fast_agent.skills.registry import SkillManifest, SkillRegistry
except ModuleNotFoundError:  # pragma: no cover - enables unit tests without fast-agent
    SkillManifest = Any
    SkillRegistry = None


def parse_skill_manifest_text(
    manifest_text: str,
    *,
    path: Path | None = None,
) -> tuple[SkillManifest | None, str | None]:
    """Parse a SkillManifest from raw SKILL.md content.

    Args:
        manifest_text: Raw SKILL.md content (frontmatter + body).
        path: Optional path for provenance/logging (defaults to in-memory).

    Returns:
        Tuple of (SkillManifest | None, error message | None).
    """
    if SkillRegistry is None:
        return None, "fast-agent-mcp is required to parse skill manifests."
    return SkillRegistry.parse_manifest_text(manifest_text, path=path)
