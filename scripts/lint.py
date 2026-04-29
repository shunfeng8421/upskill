from __future__ import annotations

import argparse
import subprocess
import sys

DEFAULT_PATHS = ["src", "tests", "scripts"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ruff lint checks for the repo.")
    parser.add_argument("--fix", action="store_true", help="Apply safe ruff fixes.")
    parser.add_argument("paths", nargs="*", default=DEFAULT_PATHS, help="Optional paths to lint.")
    args = parser.parse_args()

    command = ["ruff", "check", *args.paths]
    if args.fix:
        command.insert(2, "--fix")

    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError:
        print("Error: `ruff` is not installed in the current environment.", file=sys.stderr)
        return 1

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
