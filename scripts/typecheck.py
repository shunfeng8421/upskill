from __future__ import annotations

import argparse
import subprocess
import sys

DEFAULT_PATHS = ["src", "tests", "scripts"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ty type checks for the repo.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_PATHS,
        help="Optional paths to type check.",
    )
    args = parser.parse_args()

    command = ["ty", "check", *args.paths]

    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError:
        print("Error: `ty` is not installed in the current environment.", file=sys.stderr)
        return 1

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
