"""Skill manifest utilities for upskill."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.skills.registry import SkillManifest


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
    return SkillRegistry.parse_manifest_text(manifest_text, path=path)
