"""Internal executor protocol for evaluation runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from upskill.executors.contracts import ExecutionHandle, ExecutionRequest, ExecutionResult


class Executor(Protocol):
    """Internal execution interface used by evaluation orchestration."""

    async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
        """Start execution for a single request."""

    async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
        """Wait for a previously started execution and collect artifacts/results."""

    async def cancel(self, handle: ExecutionHandle) -> None:
        """Cancel a previously started execution."""
