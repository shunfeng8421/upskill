from __future__ import annotations

import argparse
import subprocess
import sys

DEFAULT_PATHS = ["src", "tests", "scripts"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ruff format for the repo.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply formatting changes instead of checking only.",
    )
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS, help="Optional paths to format.")
    args = parser.parse_args()

    command = ["ruff", "format", *args.paths]
    if not args.write:
        command.insert(2, "--check")

    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError:
        print("Error: `ruff` is not installed in the current environment.", file=sys.stderr)
        return 1

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
