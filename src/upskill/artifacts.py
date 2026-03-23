"""Helpers for evaluation artifact materialization."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upskill.executors.contracts import ExecutionRequest

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_AGENT_CARD_FILE_EXTENSIONS = {
    ".json",
    ".markdown",
    ".md",
    ".yaml",
    ".yml",
}


def sanitize_artifact_name(value: str) -> str:
    """Convert a human-facing label into a filesystem-friendly name."""
    normalized = _NON_ALNUM_RE.sub("-", value.strip().lower()).strip("-")
    return normalized or "execution"


def ensure_directory(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_workspace_relative_path(relative_path: str) -> Path:
    """Validate that a workspace file path stays within the workspace root."""
    normalized = Path(relative_path)
    if normalized.is_absolute():
        raise ValueError(f"Workspace file path must be relative: {relative_path}")
    if any(part == ".." for part in normalized.parts):
        raise ValueError(f"Workspace file path must not traverse parents: {relative_path}")
    return normalized


def materialize_workspace(workspace_dir: Path, workspace_files: dict[str, str]) -> None:
    """Write test workspace files into a preserved workspace directory."""
    ensure_directory(workspace_dir)
    for relative_path, content in workspace_files.items():
        file_path = workspace_dir / validate_workspace_relative_path(relative_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


def bundle_cards(
    source_dir: Path,
    destination_dir: Path,
) -> Path:
    """Copy the agent card bundle into the artifact directory."""
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)
    return destination_dir


def bundle_agent_card(
    source_dir: Path,
    destination_dir: Path,
    *,
    agent_name: str,
) -> Path:
    """Copy only the selected agent card plus shared non-card resources."""
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    ensure_directory(destination_dir)

    if source_dir.is_file():
        if source_dir.stem != agent_name:
            raise FileNotFoundError(
                f"Requested agent card {agent_name!r} does not match source file {source_dir.name!r}."
            )
        shutil.copy2(source_dir, destination_dir / source_dir.name)
        return destination_dir

    matched_card = False
    for item in source_dir.iterdir():
        destination = destination_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
            continue
        if item.stem == agent_name and item.suffix in _AGENT_CARD_FILE_EXTENSIONS:
            shutil.copy2(item, destination)
            matched_card = True
            continue
        if item.suffix not in _AGENT_CARD_FILE_EXTENSIONS:
            shutil.copy2(item, destination)

    if not matched_card:
        raise FileNotFoundError(
            f"Could not find an agent card named {agent_name!r} in {source_dir}."
        )
    return destination_dir


def materialize_skill_bundle(
    destination_dir: Path,
    request: ExecutionRequest,
) -> Path:
    """Create the explicit skills root for a run."""
    ensure_directory(destination_dir)
    if request.skill is not None:
        request.skill.save(destination_dir / request.skill.name)
    return destination_dir


def write_request_file(path: Path, request: ExecutionRequest) -> None:
    """Persist request metadata for debugging and provenance."""
    payload = asdict(request)
    if request.skill is not None:
        payload["skill"] = request.skill.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def copy_config_file(source: Path, destination: Path) -> Path | None:
    """Preserve the fast-agent config used for a run when one exists."""
    if not source.exists():
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination
