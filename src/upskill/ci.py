"""Scenario planning and report helpers for ``upskill ci``."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from upskill.models import CiReport, EvalManifest, EvalScenario, TestCase


def _normalize_relative_path(path: Path, root: Path) -> str:
    """Return a stable report path relative to ``root`` when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_eval_manifest(path: Path) -> EvalManifest:
    """Load a YAML or JSON CI manifest."""
    with open(path, encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle) or {}
    return EvalManifest.model_validate(payload)


def load_test_cases(path: Path) -> list[TestCase]:
    """Load test cases from YAML or JSON."""
    with open(path, encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle) or {}

    cases = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
    return [TestCase.model_validate(item) for item in cases]


def plan_ci_suite(
    manifest_path: Path,
    *,
    scope: str = "changed",
    base_ref: str = "origin/main",
    working_dir: Path | None = None,
) -> tuple[CiReport, list[EvalScenario]]:
    """Resolve scenario selection without executing the suite."""
    root = (working_dir or Path.cwd()).resolve()
    manifest = load_eval_manifest(manifest_path)

    changed_files: list[str] = []
    changed_skills: list[str] = []
    if scope == "changed":
        changed_files = resolve_changed_files(base_ref=base_ref, working_dir=root)
        changed_skills = resolve_changed_skill_dirs(changed_files, working_dir=root)

    selected_scenarios = select_scenarios(
        manifest,
        scope=scope,
        changed_skills=changed_skills,
    )

    report = CiReport(
        manifest_path=_normalize_relative_path(manifest_path, root),
        scope=scope,
        base_ref=base_ref if scope == "changed" else None,
        changed_files=changed_files,
        changed_skills=changed_skills,
        selected_scenarios=[scenario.id for scenario in selected_scenarios],
        success=True,
    )
    return report, selected_scenarios


def resolve_changed_files(*, base_ref: str, working_dir: Path) -> list[str]:
    """Return changed files for the current checkout."""
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=working_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.strip() or completed.stdout.strip() or "git diff failed"
        raise RuntimeError(error)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def resolve_changed_skill_dirs(changed_files: list[str], *, working_dir: Path) -> list[str]:
    """Find skill directories impacted by changed files."""
    changed_skills: set[str] = set()
    root = working_dir.resolve()

    for changed_file in changed_files:
        path = (working_dir / changed_file).resolve()
        current = path if path.is_dir() else path.parent
        while current != root and current != current.parent:
            if (current / "SKILL.md").exists():
                changed_skills.add(current.relative_to(root).as_posix())
                break
            current = current.parent

    return sorted(changed_skills)


def select_scenarios(
    manifest: EvalManifest,
    *,
    scope: str,
    changed_skills: list[str],
) -> list[EvalScenario]:
    """Filter manifest scenarios for the requested CI scope."""
    if scope == "all":
        return list(manifest.scenarios)

    changed = set(changed_skills)
    selected = []
    for scenario in manifest.scenarios:
        scenario_skills = {Path(skill).as_posix() for skill in scenario.skills}
        if scenario_skills & changed:
            selected.append(scenario)
    return selected


def write_ci_report(path: Path, report: CiReport) -> None:
    """Write the machine-readable CI report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def render_ci_report_markdown(report: CiReport) -> str:
    """Render a GitHub-friendly markdown summary."""
    lines = [
        "# upskill CI",
        "",
        f"- Scope: `{report.scope}`",
        f"- Manifest: `{report.manifest_path}`",
    ]
    if report.base_ref:
        lines.append(f"- Base ref: `{report.base_ref}`")
    if report.changed_skills:
        changed = ", ".join(f"`{item}`" for item in report.changed_skills)
        lines.append(f"- Changed skills: {changed}")
    if not report.scenarios:
        lines.extend(["", "No scenarios were selected."])
        return "\n".join(lines)

    for scenario in report.scenarios:
        lines.extend(
            [
                "",
                f"## {scenario.scenario_id}",
                "",
                "| Variant | Skills | Pass | Assertions | Judge | Tokens |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        variants = [scenario.bundle, *scenario.ablations]
        if scenario.baseline is not None:
            variants.append(scenario.baseline)
        for variant in variants:
            judge_value = f"{variant.judge_score:.2f}" if variant.judge_score is not None else "n/a"
            lines.append(
                "| "
                f"{variant.variant_id} | "
                f"{', '.join(variant.skills) or '(none)'} | "
                f"{'PASS' if variant.passed else 'FAIL'} | "
                f"{variant.assertions_passed}/{variant.assertions_total} | "
                f"{judge_value} | "
                f"{variant.total_tokens} |"
            )
        if scenario.contributions:
            lines.extend(
                [
                    "",
                    "| Skill | Hard Delta | Judge Delta | Passed Without Skill |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for contribution in scenario.contributions:
                judge_delta = (
                    f"{contribution.judge_score_delta:+.2f}"
                    if contribution.judge_score_delta is not None
                    else "n/a"
                )
                lines.append(
                    "| "
                    f"{contribution.skill} | "
                    f"{contribution.hard_score_delta:+.2f} | "
                    f"{judge_delta} | "
                    f"{'yes' if contribution.passed_without_skill else 'no'} |"
                )

    return "\n".join(lines)


def write_step_summary(report: CiReport) -> None:
    """Append the markdown summary to GitHub's step summary file when available."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(render_ci_report_markdown(report))
        handle.write("\n")
