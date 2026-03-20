"""Shared FastAgent wiring helpers for upskill."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upskill.models import Skill


def compose_instruction(instruction: str, skill: Skill | None) -> str:
    """Inject the skill content into an instruction when provided."""
    if not skill:
        return instruction
    return compose_instruction_bundle(instruction, [skill])


def compose_instruction_bundle(
    instruction: str,
    skills: list[Skill],
    *,
    mounted_paths: dict[str, str] | None = None,
) -> str:
    """Inject one or more skills into the evaluator instruction."""
    if not skills:
        return instruction

    sections = [instruction]
    if mounted_paths:
        path_lines = [
            f"- {skill.name}: {mounted_paths[skill.name]}"
            for skill in skills
            if skill.name in mounted_paths
        ]
        if path_lines:
            sections.append("## Mounted Skills\n" + "\n".join(path_lines))

    skill_sections = []
    for skill in skills:
        skill_sections.append(f"## Skill: {skill.name}\n\n{skill.body}")

    sections.append("\n\n".join(skill_sections))
    return "\n\n".join(part for part in sections if part)
