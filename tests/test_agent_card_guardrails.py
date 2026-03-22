from __future__ import annotations

from pathlib import Path

AGENT_CARDS_DIR = Path("src/upskill/agent_cards")


def _parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}

    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return {}

    frontmatter_lines: list[str] = []
    for line in lines[1:]:
        if line == "---":
            break
        frontmatter_lines.append(line)

    data: dict[str, str] = {}
    for line in frontmatter_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = value.strip()

    return data


def test_evaluator_card_does_not_pin_skills_dir() -> None:
    """Evaluation skill loading should come from --skills-dir, not card frontmatter."""
    frontmatter = _parse_frontmatter(AGENT_CARDS_DIR / "evaluator.md")
    assert "skills" not in frontmatter, (
        "evaluator.md should not define `skills:` in frontmatter. "
        "Evaluation availability must be controlled by the executor's --skills-dir."
    )
