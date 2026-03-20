"""Scenario-based CI evaluation for upskill."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from upskill.evaluate import judge_test_result, run_test_with_skills, summarize_test_results
from upskill.logging import (
    aggregate_conversation_stats,
    create_batch_folder,
    create_run_folder,
    write_batch_summary,
    write_run_metadata,
    write_run_result,
)
from upskill.models import (
    BatchSummary,
    CiReport,
    EvalManifest,
    EvalScenario,
    RunMetadata,
    RunResult,
    ScenarioContribution,
    ScenarioReport,
    ScenarioVariantResult,
    Skill,
    TestCase,
)


def _normalize_relative_path(path: Path, root: Path) -> str:
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

    return (
        CiReport(
            manifest_path=_normalize_relative_path(manifest_path, root),
            scope=scope,
            base_ref=base_ref if scope == "changed" else None,
            changed_files=changed_files,
            changed_skills=changed_skills,
            selected_scenarios=[scenario.id for scenario in selected_scenarios],
            success=True,
        ),
        selected_scenarios,
    )


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
        scenario_skills = set(Path(skill).as_posix() for skill in scenario.skills)
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


async def _evaluate_variant(
    *,
    scenario: EvalScenario,
    variant_id: str,
    variant_type: str,
    skills: list[tuple[str, Path, Skill]],
    omitted_skill: str | None,
    test_cases: list[TestCase],
    evaluator,
    judge,
    eval_model: str | None,
    judge_model: str | None,
    working_dir: Path,
    judge_enabled: bool,
    judge_criteria: list[str] | None,
    batch_id: str,
    batch_folder: Path | None,
    run_number: int,
) -> tuple[ScenarioVariantResult, RunResult]:
    test_results = []
    mounted_skills = [(path, skill) for _, path, skill in skills]
    bundle_skills = [skill for _, _, skill in skills]
    skill_labels = [label for label, _, _ in skills]

    for test_index, test_case in enumerate(test_cases, start=1):
        test_results.append(
            await run_test_with_skills(
                test_case,
                evaluator,
                bundle_skills,
                model=eval_model,
                instance_name=(
                    f"ci ({scenario.id} {variant_id} test {test_index})"
                ),
                seed_dir=working_dir,
                mounted_skills=mounted_skills,
            )
        )

    assertions_passed, assertions_total, avg_tokens, avg_turns = summarize_test_results(
        test_results
    )
    passed = all(result.success for result in test_results)
    hard_score = assertions_passed / assertions_total if assertions_total else 0.0

    judge_score = None
    judge_summary = None
    if judge is not None and judge_enabled and passed:
        judge_results = []
        for test_index, test_result in enumerate(test_results, start=1):
            judge_results.append(
                await judge_test_result(
                    scenario.id,
                    bundle_skills,
                    test_result,
                    judge,
                    judge_model=judge_model,
                    criteria=judge_criteria,
                    instance_name=(
                        f"judge ({scenario.id} {variant_id} test {test_index})"
                    ),
                )
            )
        if judge_results:
            judge_score = sum(item.normalized_score for item in judge_results) / len(judge_results)
            judge_summary = judge_results[0].summary

    aggregated_stats = aggregate_conversation_stats(test_results)

    run_folder_path: Path | None = None
    if batch_folder is not None:
        run_folder_path = create_run_folder(batch_folder, run_number)
        run_result = RunResult(
            metadata=RunMetadata(
                model=eval_model or "",
                task=scenario.id,
                batch_id=batch_id,
                run_number=run_number,
            ),
            stats=aggregated_stats,
            passed=passed,
            assertions_passed=assertions_passed,
            assertions_total=assertions_total,
            run_type="baseline" if variant_type == "baseline" else "with_skill",
            skill_name=scenario.id,
            judge_model=judge_model,
            judge_score=judge_score,
            judge_summary=judge_summary,
            scenario_id=scenario.id,
            variant_id=variant_id,
            variant_type=variant_type,
            skills=skill_labels,
            omitted_skill=omitted_skill,
        )
        write_run_metadata(run_folder_path, run_result.metadata)
        write_run_result(run_folder_path, run_result)
    else:
        run_result = RunResult(
            metadata=RunMetadata(
                model=eval_model or "",
                task=scenario.id,
                batch_id=batch_id,
                run_number=run_number,
            ),
            stats=aggregated_stats,
            passed=passed,
            assertions_passed=assertions_passed,
            assertions_total=assertions_total,
            run_type="baseline" if variant_type == "baseline" else "with_skill",
            skill_name=scenario.id,
            judge_model=judge_model,
            judge_score=judge_score,
            judge_summary=judge_summary,
            scenario_id=scenario.id,
            variant_id=variant_id,
            variant_type=variant_type,
            skills=skill_labels,
            omitted_skill=omitted_skill,
        )

    return (
        ScenarioVariantResult(
            variant_id=variant_id,
            variant_type=variant_type,  # type: ignore[arg-type]
            skills=skill_labels,
            omitted_skill=omitted_skill,
            passed=passed,
            assertions_passed=assertions_passed,
            assertions_total=assertions_total,
            hard_score=hard_score,
            judge_score=judge_score,
            judge_summary=judge_summary,
            total_tokens=aggregated_stats.total_tokens,
            average_turns=avg_turns,
            run_folder=str(run_folder_path) if run_folder_path is not None else None,
        ),
        run_result,
    )


async def run_ci_suite(
    manifest_path: Path,
    *,
    evaluator,
    judge=None,
    scope: str = "changed",
    base_ref: str = "origin/main",
    eval_model: str | None = None,
    judge_model: str | None = None,
    working_dir: Path | None = None,
    runs_dir: Path | None = None,
) -> CiReport:
    """Execute the selected scenario suite and return a machine-readable report."""
    root = (working_dir or Path.cwd()).resolve()
    report, selected_scenarios = plan_ci_suite(
        manifest_path,
        scope=scope,
        base_ref=base_ref,
        working_dir=root,
    )

    if not selected_scenarios:
        return report

    batch_id = ""
    batch_folder: Path | None = None
    all_run_results: list[RunResult] = []
    if runs_dir is not None:
        batch_id, batch_folder = create_batch_folder(runs_dir)

    run_number = 0
    for scenario in selected_scenarios:
        tests_path = (root / scenario.tests).resolve()
        test_cases = load_test_cases(tests_path)
        loaded_skills = []
        for skill_path in scenario.skills:
            absolute_skill_path = (root / skill_path).resolve()
            loaded_skills.append(
                (
                    Path(skill_path).as_posix(),
                    absolute_skill_path,
                    Skill.load(absolute_skill_path),
                )
            )

        judge_enabled = bool(scenario.judge and scenario.judge.enabled)
        judge_criteria = scenario.judge.criteria if scenario.judge else None

        run_number += 1
        bundle_result, bundle_run = await _evaluate_variant(
            scenario=scenario,
            variant_id="bundle",
            variant_type="bundle",
            skills=loaded_skills,
            omitted_skill=None,
            test_cases=test_cases,
            evaluator=evaluator,
            judge=judge,
            eval_model=eval_model,
            judge_model=judge_model,
            working_dir=root,
            judge_enabled=judge_enabled,
            judge_criteria=judge_criteria,
            batch_id=batch_id,
            batch_folder=batch_folder,
            run_number=run_number,
        )
        all_run_results.append(bundle_run)

        ablation_results: list[ScenarioVariantResult] = []
        contributions: list[ScenarioContribution] = []
        for skill_label, _, _ in loaded_skills:
            remaining = [item for item in loaded_skills if item[0] != skill_label]
            run_number += 1
            ablation_result, ablation_run = await _evaluate_variant(
                scenario=scenario,
                variant_id=f"without-{Path(skill_label).name}",
                variant_type="ablation",
                skills=remaining,
                omitted_skill=skill_label,
                test_cases=test_cases,
                evaluator=evaluator,
                judge=judge,
                eval_model=eval_model,
                judge_model=judge_model,
                working_dir=root,
                judge_enabled=judge_enabled,
                judge_criteria=judge_criteria,
                batch_id=batch_id,
                batch_folder=batch_folder,
                run_number=run_number,
            )
            all_run_results.append(ablation_run)
            ablation_results.append(ablation_result)
            contributions.append(
                ScenarioContribution(
                    skill=skill_label,
                    hard_score_delta=bundle_result.hard_score - ablation_result.hard_score,
                    judge_score_delta=(
                        None
                        if bundle_result.judge_score is None or ablation_result.judge_score is None
                        else bundle_result.judge_score - ablation_result.judge_score
                    ),
                    passed_without_skill=ablation_result.passed,
                )
            )

        baseline_result = None
        if scenario.include_baseline:
            run_number += 1
            baseline_result, baseline_run = await _evaluate_variant(
                scenario=scenario,
                variant_id="baseline",
                variant_type="baseline",
                skills=[],
                omitted_skill=None,
                test_cases=test_cases,
                evaluator=evaluator,
                judge=judge,
                eval_model=eval_model,
                judge_model=judge_model,
                working_dir=root,
                judge_enabled=judge_enabled,
                judge_criteria=judge_criteria,
                batch_id=batch_id,
                batch_folder=batch_folder,
                run_number=run_number,
            )
            all_run_results.append(baseline_run)

        report.scenarios.append(
            ScenarioReport(
                scenario_id=scenario.id,
                skills=[label for label, _, _ in loaded_skills],
                tests_path=_normalize_relative_path(tests_path, root),
                passed=bundle_result.passed,
                bundle=bundle_result,
                ablations=ablation_results,
                baseline=baseline_result,
                contributions=contributions,
            )
        )

    report.success = all(item.passed for item in report.scenarios)

    if batch_folder is not None:
        summary = BatchSummary(
            batch_id=batch_id,
            model=eval_model or "",
            task="upskill-ci",
            total_runs=len(all_run_results),
            passed_runs=sum(1 for item in all_run_results if item.passed),
            results=all_run_results,
        )
        write_batch_summary(batch_folder, summary)

    return report
