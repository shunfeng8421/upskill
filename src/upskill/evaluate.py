"""Skill evaluation - compare agent performance with and without skills using FastAgent."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

try:
    from fast_agent import ConversationSummary
    from fast_agent.agents.llm_agent import LlmAgent
except ModuleNotFoundError:  # pragma: no cover - enables unit tests without fast-agent
    ConversationSummary = Any

    class LlmAgent:  # type: ignore[no-redef]
        pass

try:
    from fast_agent.ui.rich_progress import progress_display
except Exception:  # pragma: no cover - defensive import for older fast-agent versions
    progress_display = None

from upskill.fastagent_integration import (
    compose_instruction,
    compose_instruction_bundle,
)
from upskill.logging import extract_stats_from_summary
from upskill.models import (
    CandidateEvalResult,
    CapturedArtifact,
    ConversationStats,
    EvalResults,
    JudgeCriterionScore,
    JudgeEvaluation,
    RankedSkillBatch,
    RankedSkillResult,
    Skill,
    TestCase,
    TestResult,
)
from upskill.verifiers import run_verifiers


def _hide_progress_task(task_name: str | None) -> None:
    """Best-effort hide of a completed task from the shared progress display."""
    if not task_name or progress_display is None:
        return
    hide_task = getattr(progress_display, "hide_task", None)
    if not callable(hide_task):
        return
    try:
        hide_task(task_name)
    except Exception:
        # Progress cleanup is best-effort and should never fail evaluations.
        return

logger = logging.getLogger(__name__)

PROMPT = (
    "You are an evaluator of skills. You are given a skill and a test case. "
    "You need to evaluate the skill on the test case and return a score."
)

JUDGE_CRITERIA = (
    "instruction_quality",
    "helpfulness",
    "robustness",
    "concision",
    "generalizability",
)

MAX_CAPTURE_CHARS = 4000
WORKSPACE_IGNORE_NAMES = {
    ".git",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "runs",
}


@contextmanager
def isolated_workspace(
    base_dir: Path | None = None,
    cleanup: bool = True,
    seed_dir: Path | None = None,
) -> Generator[Path]:
    """Create an isolated workspace for a test run.

    Args:
        base_dir: Optional parent directory for the workspace
        cleanup: Whether to clean up the workspace after (default True)

    Yields:
        Path to the temporary workspace directory
    """
    workspace = tempfile.mkdtemp(dir=base_dir, prefix="upskill_run_")
    workspace_path = Path(workspace)
    try:
        if seed_dir is not None:
            _seed_workspace_from_directory(seed_dir, workspace_path)
        yield workspace_path
    finally:
        if cleanup:
            try:
                shutil.rmtree(workspace_path, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors


def _seed_workspace_from_directory(seed_dir: Path, workspace: Path) -> None:
    """Copy a seed checkout into a temporary workspace."""
    for source in seed_dir.iterdir():
        if source.name in WORKSPACE_IGNORE_NAMES:
            continue
        destination = workspace / source.name
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*WORKSPACE_IGNORE_NAMES),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _copy_skill_directory(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def _capture_artifacts(test_case: TestCase, workspace: Path | None) -> list[CapturedArtifact]:
    if workspace is None or not test_case.output_file:
        return []

    target = workspace / test_case.output_file
    if not target.exists() or not target.is_file():
        return []

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [
            CapturedArtifact(
                path=test_case.output_file,
                content="<binary artifact omitted>",
                truncated=False,
            )
        ]

    truncated = len(content) > MAX_CAPTURE_CHARS
    if truncated:
        content = content[:MAX_CAPTURE_CHARS].rstrip() + "\n..."
    return [
        CapturedArtifact(
            path=test_case.output_file,
            content=content,
            truncated=truncated,
        )
    ]


def _workspace_required_for_test(
    test_case: TestCase,
    *,
    seed_dir: Path | None,
    mounted_skills: list[tuple[Path, Skill]] | None,
) -> bool:
    if test_case.output_file or test_case.validator or test_case.verifiers:
        return True
    if seed_dir is not None:
        return True
    return bool(mounted_skills)


async def _run_test_with_evaluator(
    test_case: TestCase,
    evaluator: LlmAgent,
    instruction: str | None,
    *,
    use_workspace: bool | None = None,
    instance_name: str | None = None,
    seed_dir: Path | None = None,
    mounted_skills: list[tuple[Path, Skill]] | None = None,
) -> TestResult:
    """Run a single test case using a provided evaluator agent."""
    user_content = test_case.input
    if test_case.context and test_case.context.files:
        for filename, content in test_case.context.files.items():
            user_content += f"\n\n```{filename}\n{content}\n```"

    # Determine if we need workspace isolation
    needs_workspace = use_workspace if use_workspace is not None else _workspace_required_for_test(
        test_case,
        seed_dir=seed_dir,
        mounted_skills=mounted_skills,
    )

    async def _run_in_workspace(workspace: Path | None) -> TestResult:
        clone: LlmAgent | None = None
        try:
            if workspace is not None and mounted_skills:
                for source_dir, skill in mounted_skills:
                    try:
                        relative_path = source_dir.relative_to(seed_dir) if seed_dir else None
                    except ValueError:
                        relative_path = None
                    if relative_path is None:
                        relative_path = Path(".upskill") / "skills" / skill.name
                    _copy_skill_directory(source_dir, workspace / relative_path)

            clone = await evaluator.spawn_detached_instance(name=instance_name)
            if workspace is not None:
                enable_shell = getattr(clone, "enable_shell", None)
                shell_enabled = getattr(clone, "shell_runtime_enabled", False)
                if shell_enabled and callable(enable_shell):
                    enable_shell(working_directory=workspace)

            if instruction is None:
                clone.set_instruction("")
            else:
                clone.set_instruction(instruction)
            output = await clone.send(user_content)
            stats = ConversationStats()

            # Extract stats from agent history
            try:
                history = clone.message_history
                summary = ConversationSummary(messages=history)
                stats = extract_stats_from_summary(summary)
            except Exception as exc:
                logger.exception("Failed to extract stats from evaluator history", exc_info=exc)

            artifacts = _capture_artifacts(test_case, workspace)
            validation_result = run_verifiers(
                test_case,
                output=output or "",
                workspace=workspace,
            )
            success = validation_result.passed

            return TestResult(
                test_case=test_case,
                success=success,
                output=output,
                tokens_used=stats.total_tokens,
                turns=stats.turns,
                stats=stats,
                validation_result=validation_result,
                artifacts=artifacts,
            )
        except Exception as exc:
            return TestResult(test_case=test_case, success=False, error=str(exc))
        finally:
            if clone is not None:
                try:
                    await clone.shutdown()
                except Exception as exc:
                    logger.exception("Failed to shutdown evaluator clone", exc_info=exc)
            _hide_progress_task(instance_name)

    if needs_workspace:
        with isolated_workspace(seed_dir=seed_dir) as workspace:
            return await _run_in_workspace(workspace)
    return await _run_in_workspace(None)


async def run_test_with_skills(
    test_case: TestCase,
    evaluator: LlmAgent,
    skills: list[Skill] | None = None,
    *,
    use_workspace: bool | None = None,
    model: str | None = None,
    instance_name: str | None = None,
    seed_dir: Path | None = None,
    mounted_skills: list[tuple[Path, Skill]] | None = None,
) -> TestResult:
    """Run a single test case with a bundle of injected skills."""

    bundle = skills or []
    try:
        if model is not None:
            await evaluator.set_model(model)

        mounted_paths: dict[str, str] = {}
        if mounted_skills:
            for source_dir, skill in mounted_skills:
                try:
                    relative_path = source_dir.relative_to(seed_dir) if seed_dir else None
                except ValueError:
                    relative_path = None
                if relative_path is None:
                    relative_path = Path(".upskill") / "skills" / skill.name
                mounted_paths[skill.name] = relative_path.as_posix()

        instruction = (
            compose_instruction_bundle(
                evaluator.instruction,
                bundle,
                mounted_paths=mounted_paths or None,
            )
            if bundle
            else None
        )
        return await _run_test_with_evaluator(
            test_case,
            evaluator,
            instruction,
            use_workspace=use_workspace,
            instance_name=instance_name,
            seed_dir=seed_dir,
            mounted_skills=mounted_skills,
        )
    except Exception as exc:
        return TestResult(test_case=test_case, success=False, error=str(exc))


async def run_test(
    test_case: TestCase,
    evaluator: LlmAgent,
    skill: Skill | None,
    use_workspace: bool | None = None,
    model: str | None = None,
    instance_name: str | None = None,
) -> TestResult:
    """Run a single test case using an evaluator agent.

    Args:
        test_case: The test case to run
        evaluator: Evaluator agent to run the test case
        skill: Optional skill to inject (None for baseline)
        use_workspace: Force workspace isolation (auto-detected from test_case.validator)
        model: Model to evaluate with for this test case
        instance_name: Optional evaluator instance display name
    """

    try:
        bundle = [skill] if skill else []
        return await run_test_with_skills(
            test_case,
            evaluator,
            bundle,
            use_workspace=use_workspace,
            model=model,
            instance_name=instance_name,
        )
    except Exception as exc:
        return TestResult(test_case=test_case, success=False, error=str(exc))


async def evaluate_skill(
    skill: Skill,
    test_cases: list[TestCase],
    evaluator: LlmAgent,
    model: str | None = None,
    run_baseline: bool = True,
    show_baseline_progress: bool = False,
) -> EvalResults:
    """Evaluate a skill against test cases using FastAgent.

    Args:
        skill: The skill to evaluate
        test_cases: Test cases to run
        evaluator: Evaluator agent to run the test cases
        model: Model to evaluate on (defaults to config.eval_model)
        run_baseline: Whether to also run without the skill
        show_baseline_progress: Whether to render baseline progress output

    Returns:
        EvalResults comparing skill vs baseline
    """

    results = EvalResults(skill_name=skill.name, model=model)

    base_instruction = evaluator.instruction

    async def _run_batch(
        instruction: str | None,
        label: str,
    ) -> list[TestResult]:
        tasks = []
        for index, tc in enumerate(test_cases, start=1):
            instance_name = f"eval ({label} test {index})"
            tasks.append(
                _run_test_with_evaluator(
                    tc,
                    evaluator,
                    instruction,
                    instance_name=instance_name,
                )
            )
        return await asyncio.gather(*tasks)

    if model is not None:
        await evaluator.set_model(model)

    # Run with skill
    skill_instruction = compose_instruction(base_instruction, skill)
    results.with_skill_results = await _run_batch(skill_instruction, "with-skill")

    # Calculate with-skill metrics
    successes = sum(1 for r in results.with_skill_results if r.success)
    results.with_skill_success_rate = successes / len(test_cases) if test_cases else 0
    results.with_skill_total_tokens = sum(
        r.stats.total_tokens for r in results.with_skill_results
    )
    results.with_skill_avg_turns = (
        sum(r.stats.turns for r in results.with_skill_results) / len(test_cases)
        if test_cases
        else 0
    )

    # Run baseline if requested
    if run_baseline:
        pause_cm = nullcontext()
        if not show_baseline_progress and progress_display is not None:
            paused = getattr(progress_display, "paused", None)
            if callable(paused):
                pause_cm = paused()

        with pause_cm:
            results.baseline_results = await _run_batch(None, "baseline")

        successes = sum(1 for r in results.baseline_results if r.success)
        results.baseline_success_rate = successes / len(test_cases) if test_cases else 0
        results.baseline_total_tokens = sum(
            r.stats.total_tokens for r in results.baseline_results
        )
        results.baseline_avg_turns = (
            sum(r.stats.turns for r in results.baseline_results) / len(test_cases)
            if test_cases
            else 0
        )

    return results


def summarize_test_results(test_results: list[TestResult]) -> tuple[int, int, float, float]:
    """Return assertions passed/total plus average tokens and turns."""
    assertions_passed = 0
    assertions_total = 0
    total_tokens = 0
    total_turns = 0

    for result in test_results:
        total_tokens += result.stats.total_tokens
        total_turns += result.stats.turns
        if result.validation_result:
            assertions_passed += result.validation_result.assertions_passed
            assertions_total += result.validation_result.assertions_total
        else:
            assertions_total += 1
            if result.success:
                assertions_passed += 1

    count = len(test_results)
    avg_tokens = total_tokens / count if count else 0.0
    avg_turns = total_turns / count if count else 0.0
    return assertions_passed, assertions_total, avg_tokens, avg_turns


def build_default_judge_evaluation(summary: str) -> JudgeEvaluation:
    """Build a neutral fallback judge result."""
    return JudgeEvaluation(
        summary=summary,
        criteria=[
            JudgeCriterionScore(
                criterion=criterion,
                score=3,
                rationale="Judge output unavailable; using neutral fallback score.",
            )
            for criterion in JUDGE_CRITERIA
        ],
    )


async def judge_test_result(
    task: str,
    skill: Skill | list[Skill],
    test_result: TestResult,
    judge: LlmAgent,
    *,
    judge_model: str | None = None,
    criteria: tuple[str, ...] | list[str] | None = None,
    instance_name: str | None = None,
) -> JudgeEvaluation:
    """Run LLM-as-a-judge for one executed candidate/test result."""

    clone: LlmAgent | None = None
    try:
        clone = await judge.spawn_detached_instance(name=instance_name)
        if judge_model is not None:
            await clone.set_model(judge_model)

        skill_bundle = skill if isinstance(skill, list) else [skill]
        artifact_sections = []
        for artifact in test_result.artifacts:
            artifact_sections.append(
                f"Artifact path: {artifact.path}\n"
                f"Artifact content:\n{artifact.content}"
            )
        artifact_block = "\n\n".join(artifact_sections) if artifact_sections else "none"
        rubric = tuple(criteria or JUDGE_CRITERIA)
        skill_sections = []
        for item in skill_bundle:
            skill_sections.append(
                f"Skill name: {item.name}\n"
                f"Skill description: {item.description}\n"
                f"Skill body:\n{item.body}"
            )
        verifier_payload = [
            spec.model_dump(mode="json") for spec in test_result.test_case.effective_verifiers()
        ]
        validation_payload = (
            test_result.validation_result.model_dump(mode="json")
            if test_result.validation_result
            else "none"
        )

        prompt = (
            f"Original task:\n{task}\n\n"
            f"Candidate skill bundle:\n{chr(10).join(skill_sections)}\n\n"
            f"Test case input:\n{test_result.test_case.input}\n\n"
            f"Verifiers:\n{verifier_payload}\n\n"
            f"Agent output:\n{test_result.output or ''}\n\n"
            f"Captured artifacts:\n{artifact_block}\n\n"
            f"Execution success: {test_result.success}\n"
            f"Execution error: {test_result.error or ''}\n"
            f"Validation result: {validation_payload}\n\n"
            "Score this executed candidate against the rubric. "
            "Return structured data with exactly these criteria: "
            f"{', '.join(rubric)}."
        )
        result, _ = await clone.structured(prompt, JudgeEvaluation)
        if result is None:
            return build_default_judge_evaluation("Judge returned no structured result.")
        return result
    except Exception as exc:
        logger.exception("Judge evaluation failed", exc_info=exc)
        return build_default_judge_evaluation(f"Judge evaluation failed: {exc}")
    finally:
        if clone is not None:
            try:
                await clone.shutdown()
            except Exception as exc:
                logger.exception("Failed to shutdown judge clone", exc_info=exc)
        _hide_progress_task(instance_name)


def rank_candidate_results(
    task: str,
    candidate_results: list[CandidateEvalResult],
    *,
    skill_generation_model: str,
    evaluation_model: str,
    judge_model: str | None,
    judge_strategy: str,
    tests: list[TestCase],
    judge_weight: float = 0.3,
) -> RankedSkillBatch:
    """Rank evaluated candidates using hard score, judge score, and token efficiency."""

    if not candidate_results:
        return RankedSkillBatch(
            task=task,
            skill_generation_model=skill_generation_model,
            evaluation_model=evaluation_model,
            judge_model=judge_model,
            judge_strategy=judge_strategy,
            candidate_count=0,
            tests=tests,
        )

    min_avg_tokens = min(result.average_tokens for result in candidate_results)
    max_avg_tokens = max(result.average_tokens for result in candidate_results)
    token_range = max_avg_tokens - min_avg_tokens

    for result in candidate_results:
        if token_range <= 0:
            result.token_efficiency_score = 1.0
        else:
            result.token_efficiency_score = 1 - (
                (result.average_tokens - min_avg_tokens) / token_range
            )

        hard_score = result.hard_score
        judge_score = result.judge_score
        token_score = result.token_efficiency_score
        result.composite_score = (
            0.6 * hard_score
            + judge_weight * judge_score
            + 0.1 * token_score
        )

    best_hard_score = max(result.hard_score for result in candidate_results)
    hard_gate_threshold = max(0.0, best_hard_score - 0.2)
    for result in candidate_results:
        result.hard_gate_failed = result.hard_score < hard_gate_threshold

    ordered = sorted(
        candidate_results,
        key=lambda result: (
            result.hard_gate_failed,
            -result.hard_score,
            -result.judge_score,
            -result.token_efficiency_score,
            -result.composite_score,
            result.candidate_id,
        ),
    )

    ranked_results: list[RankedSkillResult] = []
    for index, result in enumerate(ordered, start=1):
        result.skill.metadata.candidate_id = result.candidate_id
        summary = None
        if result.judge_evaluations:
            summaries = [item.summary for item in result.judge_evaluations if item.summary]
            summary = summaries[0] if summaries else None
        margin = None
        if index < len(ordered):
            margin = result.composite_score - ordered[index].composite_score
        ranked_results.append(
            RankedSkillResult(
                rank=index,
                candidate=result,
                judge_model=judge_model,
                judge_summary=summary,
                score_margin_from_next=margin,
            )
        )

    return RankedSkillBatch(
        task=task,
        skill_generation_model=skill_generation_model,
        evaluation_model=evaluation_model,
        judge_model=judge_model,
        judge_strategy=judge_strategy,
        candidate_count=len(candidate_results),
        ranked_results=ranked_results,
        tests=tests,
    )


async def evaluate_skill_candidates(
    task: str,
    candidates: list[Skill],
    test_cases: list[TestCase],
    evaluator: LlmAgent,
    judge: LlmAgent | None,
    *,
    skill_generation_model: str | None = None,
    evaluation_model: str,
    judge_model: str | None,
    judge_strategy: str = "pointwise",
    judge_weight: float = 0.3,
) -> RankedSkillBatch:
    """Evaluate and rank multiple candidate skills."""

    if judge_strategy != "pointwise":
        raise ValueError("Only pointwise judge strategy is supported in v1.")

    candidate_results: list[CandidateEvalResult] = []
    for index, skill in enumerate(candidates, start=1):
        eval_results = await evaluate_skill(
            skill,
            test_cases=test_cases,
            evaluator=evaluator,
            model=evaluation_model,
            run_baseline=False,
            show_baseline_progress=False,
        )
        assertions_passed, assertions_total, avg_tokens, avg_turns = summarize_test_results(
            eval_results.with_skill_results
        )
        hard_score = (
            assertions_passed / assertions_total if assertions_total else 0.0
        )

        judge_evaluations: list[JudgeEvaluation] = []
        if judge is not None:
            for test_index, test_result in enumerate(eval_results.with_skill_results, start=1):
                judge_evaluations.append(
                    await judge_test_result(
                        task,
                        skill,
                        test_result,
                        judge,
                        judge_model=judge_model,
                        instance_name=(
                            f"judge ({skill.metadata.candidate_id or index} test {test_index})"
                        ),
                    )
                )

        judge_score = (
            sum(item.normalized_score for item in judge_evaluations) / len(judge_evaluations)
            if judge_evaluations
            else 0.0
        )
        candidate_id = skill.metadata.candidate_id or f"candidate-{index}"
        candidate_results.append(
            CandidateEvalResult(
                candidate_id=candidate_id,
                skill=skill,
                test_results=eval_results.with_skill_results,
                judge_evaluations=judge_evaluations,
                assertions_passed=assertions_passed,
                assertions_total=assertions_total,
                hard_score=hard_score,
                judge_score=judge_score,
                average_tokens=avg_tokens,
                average_turns=avg_turns,
            )
        )

    return rank_candidate_results(
        task,
        candidate_results,
        skill_generation_model=skill_generation_model or evaluation_model,
        evaluation_model=evaluation_model,
        judge_model=judge_model,
        judge_strategy=judge_strategy,
        tests=test_cases,
        judge_weight=judge_weight,
    )


def get_failure_descriptions(results: EvalResults) -> list[str]:
    """Extract descriptions of failed tests for refinement."""
    failures = []
    for result in results.with_skill_results:
        if not result.success:
            desc = f"Input: {result.test_case.input}"
            if result.error:
                desc += f" | Error: {result.error}"
            elif result.output:
                desc += f" | Output: {result.output[:200]}..."
            if result.test_case.expected:
                desc += f" | Expected: {result.test_case.expected}"
            failures.append(desc)
    return failures
