from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.cpd import build_cpd_command, resolve_cli_exit_code, resolve_platform

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_platform_maps_common_linux_labels() -> None:
    platform_config = resolve_platform(system="linux", arch="x86_64")

    assert platform_config.system == "linux"
    assert platform_config.arch == "x86_64"
    assert platform_config.os_label == "linux"
    assert platform_config.arch_label == "x64"
    assert platform_config.java_name == "java"
    assert platform_config.pmd_name == "pmd"


def test_build_cpd_command_includes_expected_arguments(tmp_path: Path) -> None:
    platform_config = resolve_platform(system="linux", arch="x86_64")
    pmd_dir = tmp_path / "pmd-bin"
    src_dir = tmp_path / "src"
    excluded_path = src_dir / "skip_me.py"

    command = build_cpd_command(
        platform_config=platform_config,
        pmd_dir=pmd_dir,
        src_dir=src_dir,
        excluded_paths=[excluded_path],
        min_tokens=120,
        output_format="xml",
    )

    assert command == [
        str(pmd_dir / "bin" / "pmd"),
        "cpd",
        "--language",
        "python",
        "--minimum-tokens",
        "120",
        "--dir",
        str(src_dir),
        "--format",
        "xml",
        "--exclude",
        str(excluded_path),
    ]


def test_resolve_cli_exit_code_honors_check_mode() -> None:
    assert resolve_cli_exit_code(cpd_exit_code=0, check=False) == 0
    assert resolve_cli_exit_code(cpd_exit_code=4, check=False) == 0
    assert resolve_cli_exit_code(cpd_exit_code=4, check=True) == 1
    assert resolve_cli_exit_code(cpd_exit_code=7, check=True) == 7
