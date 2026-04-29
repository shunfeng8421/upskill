"""Parse fast-agent result artifacts into upskill-friendly data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fast_agent import ConversationSummary
from fast_agent.mcp.prompt_serialization import load_messages

from upskill.logging import extract_stats_from_summary

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from upskill.models import ConversationStats


@dataclass(slots=True, frozen=True)
class ParsedExecutionResult:
    """Parsed view of a fast-agent result export."""

    output_text: str | None
    stats: ConversationStats


def _extract_output_text(messages: Sequence[object]) -> str | None:
    for message in reversed(messages):
        role = getattr(message, "role", None)
        if role != "assistant":
            continue
        last_text = getattr(message, "last_text", None)
        if callable(last_text):
            text = last_text()
            if text:
                return text
    return None


def parse_fast_agent_results(results_path: Path) -> ParsedExecutionResult:
    """Load and summarize a fast-agent JSON history export."""
    messages = load_messages(str(results_path))
    summary = ConversationSummary(messages=messages)
    return ParsedExecutionResult(
        output_text=_extract_output_text(messages),
        stats=extract_stats_from_summary(summary),
    )
