from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from upskill.models import ExpectedSpec, ValidationResult, VerifierSpec
from upskill.models import TestCase as UpskillTestCase
from upskill.validators import register_validator
from upskill.verifiers import run_verifiers

if TYPE_CHECKING:
    from pathlib import Path


@register_validator("test-counting-validator")
def _test_counting_validator(
    workspace: Path,
    output_file: str,
    **_: object,
) -> ValidationResult:
    target = workspace / output_file
    passed = target.exists()
    return ValidationResult(
        passed=passed,
        assertions_passed=2 if passed else 0,
        assertions_total=2,
        error_message=None if passed else f"missing file: {output_file}",
    )


def test_run_verifiers_supports_legacy_expected_contains() -> None:
    test_case = UpskillTestCase(
        input="say hello",
        expected=ExpectedSpec(contains=["hello", "world"]),
    )

    result = run_verifiers(test_case, output="Hello, world!", workspace=None)

    assert result.passed is True
    assert result.assertions_passed == 1
    assert result.assertions_total == 1


def test_run_verifiers_supports_file_verifiers(tmp_path: Path) -> None:
    target = tmp_path / "report.txt"
    target.write_text("bundle ok", encoding="utf-8")
    test_case = UpskillTestCase(
        input="write file",
        verifiers=[
            VerifierSpec(type="file_exists", path="report.txt"),
            VerifierSpec(type="file_contains", path="report.txt", text="bundle ok"),
        ],
    )

    result = run_verifiers(test_case, output="", workspace=tmp_path)

    assert result.passed is True
    assert result.assertions_passed == 2
    assert result.assertions_total == 2


def test_run_verifiers_supports_command_verifier(tmp_path: Path) -> None:
    script = tmp_path / "check.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    test_case = UpskillTestCase(
        input="run assertion script",
        verifiers=[VerifierSpec(type="command", cmd=f"'{sys.executable}' check.py")],
    )

    result = run_verifiers(test_case, output="", workspace=tmp_path)

    assert result.passed is True
    assert result.assertions_passed == 1


def test_run_verifiers_translates_legacy_validator(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    target.write_text("ok", encoding="utf-8")
    test_case = UpskillTestCase(
        input="validate artifact",
        validator="test-counting-validator",
        output_file="artifact.txt",
    )

    result = run_verifiers(test_case, output="", workspace=tmp_path)

    assert result.passed is True
    assert result.assertions_passed == 2
    assert result.assertions_total == 2


def test_run_verifiers_reports_failures(tmp_path: Path) -> None:
    test_case = UpskillTestCase(
        input="write report",
        verifiers=[
            VerifierSpec(type="file_exists", path="report.txt"),
            VerifierSpec(
                type="command",
                cmd=f"'{sys.executable}' -c 'import sys; sys.exit(1)'",
            ),
        ],
    )

    result = run_verifiers(test_case, output="", workspace=tmp_path)

    assert result.passed is False
    assert result.assertions_passed == 0
    assert result.assertions_total == 2
    assert result.error_message is not None


def test_test_case_rejects_missing_expectation_configuration() -> None:
    with pytest.raises(
        ValueError, match="requires at least one of expected, verifiers, or validator"
    ):
        UpskillTestCase(input="missing all checks")
