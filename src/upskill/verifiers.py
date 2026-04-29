"""Deterministic verifier execution for upskill test cases."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from upskill.models import TestCase, ValidationResult, VerifierSpec
from upskill.validators import get_validator

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
MAX_COMMAND_OUTPUT_CHARS = 1200


def _build_validation_result(
    passed: bool,
    *,
    error_message: str | None = None,
    details: list[str] | None = None,
) -> ValidationResult:
    return ValidationResult(
        passed=passed,
        assertions_passed=1 if passed else 0,
        assertions_total=1,
        error_message=error_message,
        details=details or [],
    )


def _format_command_failure(output: str) -> str:
    compact = output.strip()
    if len(compact) > MAX_COMMAND_OUTPUT_CHARS:
        compact = compact[:MAX_COMMAND_OUTPUT_CHARS].rstrip() + "..."
    return compact or "command exited with a non-zero status"


def _resolve_values(spec: VerifierSpec) -> list[str]:
    if spec.values:
        return spec.values
    if spec.text:
        return [spec.text]
    return []


def _run_contains_verifier(spec: VerifierSpec, output: str) -> ValidationResult:
    required = [value for value in _resolve_values(spec) if value.strip()]
    if not required:
        return _build_validation_result(False, error_message="contains verifier is missing values")

    output_lower = output.lower()
    missing = [item for item in required if item.lower() not in output_lower]
    if missing:
        return _build_validation_result(
            False,
            error_message=f"missing required output text: {missing[0]}",
            details=[f"missing: {item}" for item in missing],
        )
    return _build_validation_result(True)


def _require_workspace(spec: VerifierSpec, workspace: Path | None) -> ValidationResult | None:
    if workspace is not None:
        return None
    return _build_validation_result(
        False,
        error_message=f"{spec.type} verifier requires a workspace",
    )


def _run_file_exists_verifier(spec: VerifierSpec, workspace: Path | None) -> ValidationResult:
    workspace_error = _require_workspace(spec, workspace)
    if workspace_error is not None:
        return workspace_error
    assert workspace is not None
    if not spec.path:
        return _build_validation_result(False, error_message="file_exists verifier is missing path")

    target = workspace / spec.path
    if target.exists():
        return _build_validation_result(True)
    return _build_validation_result(
        False,
        error_message=f"expected file does not exist: {spec.path}",
    )


def _run_file_contains_verifier(spec: VerifierSpec, workspace: Path | None) -> ValidationResult:
    workspace_error = _require_workspace(spec, workspace)
    if workspace_error is not None:
        return workspace_error
    assert workspace is not None
    if not spec.path:
        return _build_validation_result(
            False,
            error_message="file_contains verifier is missing path",
        )

    target = workspace / spec.path
    if not target.exists():
        return _build_validation_result(
            False,
            error_message=f"expected file does not exist: {spec.path}",
        )

    required = [value for value in _resolve_values(spec) if value.strip()]
    if not required:
        return _build_validation_result(
            False,
            error_message="file_contains verifier is missing text or values",
        )

    content = target.read_text(encoding="utf-8")
    content_lower = content.lower()
    missing = [item for item in required if item.lower() not in content_lower]
    if missing:
        return _build_validation_result(
            False,
            error_message=f"missing required file text: {missing[0]}",
            details=[f"missing: {item}" for item in missing],
        )
    return _build_validation_result(True)


def _run_command_verifier(spec: VerifierSpec, workspace: Path | None) -> ValidationResult:
    workspace_error = _require_workspace(spec, workspace)
    if workspace_error is not None:
        return workspace_error
    assert workspace is not None
    if not spec.cmd:
        return _build_validation_result(False, error_message="command verifier is missing cmd")

    timeout_seconds = DEFAULT_COMMAND_TIMEOUT_SECONDS
    if spec.config and "timeout_seconds" in spec.config:
        timeout_seconds = int(spec.config["timeout_seconds"])

    completed = subprocess.run(
        spec.cmd,
        shell=True,
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode == 0:
        return _build_validation_result(True)

    combined_output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return _build_validation_result(
        False,
        error_message=_format_command_failure(combined_output),
    )


def _run_legacy_validator_verifier(spec: VerifierSpec, workspace: Path | None) -> ValidationResult:
    workspace_error = _require_workspace(spec, workspace)
    if workspace_error is not None:
        return workspace_error
    assert workspace is not None
    if not spec.name:
        return _build_validation_result(False, error_message="validator verifier is missing name")

    validator = get_validator(spec.name)
    if validator is None:
        return _build_validation_result(
            False,
            error_message=f"unknown validator: {spec.name}",
        )

    config = spec.config or {}
    return validator(
        workspace=workspace,
        output_file=spec.path or "",
        **config,
    )


def run_verifier(
    spec: VerifierSpec,
    *,
    output: str,
    workspace: Path | None,
) -> ValidationResult:
    """Run one verifier against the current output/workspace."""

    if spec.type == "contains":
        return _run_contains_verifier(spec, output)
    if spec.type == "file_exists":
        return _run_file_exists_verifier(spec, workspace)
    if spec.type == "file_contains":
        return _run_file_contains_verifier(spec, workspace)
    if spec.type == "command":
        return _run_command_verifier(spec, workspace)
    if spec.type == "validator":
        return _run_legacy_validator_verifier(spec, workspace)

    return _build_validation_result(
        False,
        error_message=f"unsupported verifier type: {spec.type}",
    )


def run_verifiers(
    test_case: TestCase,
    *,
    output: str,
    workspace: Path | None,
) -> ValidationResult:
    """Run all verifiers configured for a test case."""

    specs = test_case.effective_verifiers()
    if not specs:
        return ValidationResult(
            passed=False,
            assertions_passed=0,
            assertions_total=0,
            error_message="no verifiers configured",
        )

    passed = 0
    total = 0
    details: list[str] = []
    error_messages: list[str] = []

    for spec in specs:
        result = run_verifier(spec, output=output, workspace=workspace)
        passed += result.assertions_passed
        total += result.assertions_total
        if result.error_message:
            error_messages.append(result.error_message)
        if result.details:
            details.extend(result.details)

    return ValidationResult(
        passed=passed == total,
        assertions_passed=passed,
        assertions_total=total,
        error_message="; ".join(error_messages) if error_messages else None,
        details=details,
    )
