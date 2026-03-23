from __future__ import annotations

from scripts.check import build_check_steps


def test_build_check_steps_includes_cpd_and_pytest() -> None:
    steps = build_check_steps()

    assert [step.name for step in steps] == ["format", "lint", "typecheck", "cpd", "pytest"]
    assert steps[3].command[-1] == "--check"
    assert steps[4].command[1:] == ("-m", "pytest", "-v")


def test_build_check_steps_can_skip_pytest() -> None:
    steps = build_check_steps(skip_tests=True)

    assert [step.name for step in steps] == ["format", "lint", "typecheck", "cpd"]
