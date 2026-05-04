"""Helpers for exporting fast-agent traces to artifact datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from upskill.artifacts import sanitize_artifact_name

if TYPE_CHECKING:
    from upskill.executors.contracts import ExecutionRequest


def build_trace_dataset_path(request: ExecutionRequest, *, run_id: str | None = None) -> str:
    """Build a stable, identifiable dataset path for a fast-agent trace export."""
    operation = request.metadata.get("operation")
    operation_name = operation if isinstance(operation, str) and operation else "eval"
    trace_id = run_id or uuid4().hex[:12]
    label = sanitize_artifact_name(request.label)
    model = sanitize_artifact_name(request.model)
    filename = f"{sanitize_artifact_name(f'{trace_id}-{label}')}.jsonl"
    return f"traces/{sanitize_artifact_name(operation_name)}/{model}/{filename}"
