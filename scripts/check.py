from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CheckStep:
    """A named local quality-gate command."""

    name: str
    command: tuple[str, ...]


def build_check_steps(*, skip_tests: bool = False) -> list[CheckStep]:
    """Build the local quality-gate command sequence."""
    python_executable = sys.executable
    steps = [
        CheckStep("format", (python_executable, str(PROJECT_ROOT / "scripts" / "format.py"))),
        CheckStep("lint", (python_executable, str(PROJECT_ROOT / "scripts" / "lint.py"))),
        CheckStep("typecheck", (python_executable, str(PROJECT_ROOT / "scripts" / "typecheck.py"))),
        CheckStep(
            "cpd",
            (python_executable, str(PROJECT_ROOT / "scripts" / "cpd.py"), "--check"),
        ),
    ]
    if not skip_tests:
        steps.append(CheckStep("pytest", (python_executable, "-m", "pytest", "-v")))
    return steps


def run_step(step: CheckStep) -> int:
    """Run a single quality-gate step."""
    print(f"\n==> {step.name}: {' '.join(step.command)}", flush=True)
    try:
        completed = subprocess.run(step.command, cwd=PROJECT_ROOT, check=False)
    except FileNotFoundError as error:
        print(f"Error: failed to execute {step.name}: {error}", file=sys.stderr)
        return 1
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local quality-gate sequence.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest after running the static checks.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run `uv sync --extra dev` before the quality gates.",
    )
    args = parser.parse_args()

    if args.sync:
        sync_step = CheckStep("sync", ("uv", "sync", "--extra", "dev"))
        if run_step(sync_step) != 0:
            return 1

    for step in build_check_steps(skip_tests=args.skip_tests):
        if run_step(step) != 0:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
