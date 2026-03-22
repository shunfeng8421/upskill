"""CLI interface for upskill."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, TypeVar, cast

import click
from dotenv import load_dotenv
from fast_agent import FastAgent
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from upskill.config import Config, resolve_upskill_config_path
from upskill.evaluate import build_eval_requests, evaluate_skill, get_failure_descriptions
from upskill.executors.local_fast_agent import LocalFastAgentExecutor
from upskill.executors.remote_fast_agent import RemoteFastAgentExecutor
from upskill.generate import generate_skill, generate_tests, improve_skill, refine_skill
from upskill.hf_jobs import JobsConfig
from upskill.logging import (
    aggregate_conversation_stats,
    create_batch_folder,
    create_run_folder,
    load_batch_summary,
    load_run_result,
    summarize_runs_to_csv,
    write_batch_summary,
    write_run_metadata,
    write_run_result,
)
from upskill.model_resolution import ResolvedModels, resolve_models
from upskill.models import (
    BatchSummary,
    EvalResults,
    RunMetadata,
    RunResult,
    Skill,
    SkillRecord,
    TestCase,
    TestResult,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fast_agent.agents.llm_agent import LlmAgent
    from fast_agent.interfaces import AgentProtocol

    from upskill.executors.base import Executor

load_dotenv()

console = Console()


class FastAgentSession(Protocol):
    """Typed view of the loaded fast-agent session used by upskill."""

    skill_gen: AgentProtocol
    test_gen: AgentProtocol
    evaluator: LlmAgent


EvalPlotLabelField = Literal["model", "skill_name"]
ExecutorName = Literal["local", "jobs"]
CommandFunction = TypeVar("CommandFunction", bound=Callable[..., object])


def _jobs_execution_options(
    *,
    executor_help: str,
    runs_dir_help: str,
) -> Callable[[CommandFunction], CommandFunction]:
    """Attach the shared remote-execution CLI options to a command."""
    options = (
        click.option(
            "--executor",
            type=click.Choice(["local", "jobs"]),
            default="local",
            show_default=True,
            help=executor_help,
        ),
        click.option("--artifact-repo", help="Dataset repo for remote fast-agent job artifacts"),
        click.option(
            "--wait/--no-wait",
            default=False,
            help="Wait for remote fast-agent jobs and download results",
        ),
        click.option(
            "--jobs-timeout",
            default="2h",
            show_default=True,
            help="HF Jobs timeout for remote fast-agent runs",
        ),
        click.option(
            "--jobs-flavor",
            default="cpu-basic",
            show_default=True,
            help="HF Jobs hardware flavor for remote fast-agent runs",
        ),
        click.option(
            "--jobs-secrets",
            default="HF_TOKEN",
            show_default=True,
            help="Comma-separated HF Job secrets to forward",
        ),
        click.option("--jobs-namespace", help="Optional Hugging Face Jobs namespace"),
        click.option(
            "--max-parallel",
            type=click.IntRange(min=1),
            default=5,
            show_default=True,
            help="Maximum concurrent evaluation executions per phase",
        ),
        click.option("--runs-dir", type=click.Path(), help=runs_dir_help),
        click.option(
            "--log-runs/--no-log-runs", default=True, help="Log run data (default: enabled)"
        ),
    )

    def decorator(function: CommandFunction) -> CommandFunction:
        wrapped = function
        for option in reversed(options):
            wrapped = option(wrapped)
        return wrapped

    return decorator


@asynccontextmanager
async def _fast_agent_context(config: Config | None = None) -> AsyncIterator[FastAgentSession]:
    config = config or Config.load()
    fast = FastAgent(
        "upskill",
        config_path=str(config.effective_fastagent_config),
        ignore_unknown_args=True,
        parse_cli_args=False,
    )

    @fast.agent()
    async def empty() -> None:
        pass

    cards = resources.files("upskill").joinpath("agent_cards")
    with resources.as_file(cards) as cards_path:
        fast.load_agents(cards_path)

    async with fast.run() as agent:
        yield cast("FastAgentSession", agent)


async def _set_agent_model(agent: object, model: str | None) -> None:
    """Best-effort model assignment for a fast-agent instance."""
    if not model:
        return
    set_model = getattr(agent, "set_model", None)
    if not callable(set_model):
        return
    result = set_model(model)
    if inspect.isawaitable(result):
        await result


def _require_resolved_model(value: str | None, *, field: str, command: str) -> str:
    """Require a non-null resolved model value for a command."""
    if value is None:
        raise RuntimeError(
            f"Model resolution bug: `{command}` requires resolved `{field}` to be set."
        )
    return value


def _require_resolved_models(values: list[str], *, field: str, command: str) -> list[str]:
    """Require a non-empty resolved model list for a command."""
    if not values:
        raise RuntimeError(
            f"Model resolution bug: `{command}` requires resolved `{field}` to be non-empty."
        )
    return values


def _require_path(value: Path | None, *, field: str, command: str) -> Path:
    """Require a resolved filesystem path for logging flows."""
    if value is None:
        raise RuntimeError(f"Internal bug: `{command}` requires `{field}` to be set.")
    return value


def _build_executor(
    name: ExecutorName,
    *,
    jobs_config: JobsConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> Executor:
    """Construct an evaluation executor from a user-facing executor name."""
    if name == "local":
        return LocalFastAgentExecutor()
    if jobs_config is None:
        raise click.ClickException("The jobs executor requires jobs configuration.")
    return RemoteFastAgentExecutor(
        jobs_config=jobs_config,
        progress_callback=progress_callback,
    )


def _require_jobs_config(
    *,
    executor_name: ExecutorName,
    artifact_repo: str | None,
    wait: bool,
    jobs_timeout: str,
    jobs_flavor: str,
    jobs_secrets: str,
    jobs_namespace: str | None,
) -> JobsConfig | None:
    """Build jobs config when the jobs executor is selected."""
    if executor_name != "jobs":
        return None
    if not artifact_repo:
        raise click.ClickException("--artifact-repo is required when using --executor jobs.")
    return JobsConfig(
        artifact_repo=artifact_repo,
        wait=wait,
        jobs_timeout=jobs_timeout,
        jobs_flavor=jobs_flavor,
        jobs_secrets=jobs_secrets,
        jobs_namespace=jobs_namespace,
    )


def _print_model_plan(command: str, resolved: ResolvedModels, runs: int | None = None) -> None:
    """Print resolved model plan for command execution."""
    console.print("[dim]Resolved model plan:[/dim]")

    if command == "generate":
        console.print(f"  Skill Generation Model: {resolved.skill_generation_model}")
        console.print(f"  Test Generation Model: {resolved.test_generation_model}")
        console.print(f"  Evaluation Model (main loop): {resolved.skill_generation_model}")
        if resolved.extra_eval_model:
            console.print(f"  Evaluation Model (extra pass): {resolved.extra_eval_model}")
        return

    if command in {"eval", "benchmark"}:
        models = ", ".join(resolved.evaluation_models)
        console.print(f"  Evaluation Model(s): {models}")
        if runs is not None:
            console.print(f"  Runs per model: {runs}")
        baseline_state = (
            "off (benchmark mode)"
            if resolved.is_benchmark_mode
            else ("on" if resolved.run_baseline else "off")
        )
        console.print(f"  Baseline: {baseline_state}")
        console.print(f"  Test Generation Model: {resolved.test_generation_model}")


def _render_bar(value: float, width: int = 20) -> str:
    """Render a simple text bar for a 0-1 value."""
    if width <= 0:
        return ""
    clamped = max(0.0, min(1.0, value))
    filled = round(clamped * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _print_eval_progress(message: str) -> None:
    """Render a lightweight evaluation progress line."""
    console.print(f"[dim]{message}[/dim]")


class EvalPlotResult(TypedDict):
    """Structured plot data for eval runs."""

    model: str
    skill_name: str
    with_skill_rate: float
    with_skill_tokens: int
    baseline_rate: float
    baseline_tokens: int
    has_baseline: bool


def _success_rate(run: RunResult) -> float:
    """Compute a success rate from a run result."""
    if run.assertions_total == 0:
        return 0.0
    return run.assertions_passed / run.assertions_total


def _select_baseline_run(
    baseline_runs: list[RunResult],
    with_skill_run: RunResult,
) -> RunResult | None:
    """Select the most relevant baseline run for a with-skill run."""
    if not baseline_runs:
        return None
    with_number = with_skill_run.metadata.run_number
    eligible = [run for run in baseline_runs if run.metadata.run_number <= with_number]
    if eligible:
        return eligible[-1]
    return baseline_runs[-1]


def _build_logged_run_result(
    *,
    model: str,
    task: str,
    batch_id: str,
    run_number: int,
    test_results: list[TestResult],
    assertions_total: int,
    passed: bool,
    run_type: str,
    skill_name: str,
) -> RunResult:
    """Construct a persisted run summary from reconstructed test results."""
    assertions_passed = 0
    computed_assertions_total = 0
    for result in test_results:
        if result.validation_result is not None:
            assertions_passed += result.validation_result.assertions_passed
            computed_assertions_total += result.validation_result.assertions_total
            continue

        assertions_passed += int(result.success)
        computed_assertions_total += 1

    return RunResult(
        metadata=RunMetadata(
            model=model,
            task=task,
            batch_id=batch_id,
            run_number=run_number,
        ),
        stats=aggregate_conversation_stats(test_results),
        passed=passed,
        assertions_passed=assertions_passed,
        assertions_total=computed_assertions_total or assertions_total,
        run_type=run_type,
        skill_name=skill_name,
    )


def _persist_logged_run(run_folder: Path, run_result: RunResult) -> None:
    """Write the standard metadata and result files for a run."""
    write_run_metadata(run_folder, run_result.metadata)
    write_run_result(run_folder, run_result)


def _persist_comparison_run_results(
    *,
    batch_folder: Path,
    model: str,
    task: str,
    batch_id: str,
    first_run_number: int,
    results: EvalResults,
    assertions_total: int,
    run_baseline: bool,
    with_skill_passed: bool,
    skill_name: str,
) -> list[RunResult]:
    """Persist baseline/with-skill summaries for one evaluation pass."""
    persisted_results: list[RunResult] = []
    run_number = first_run_number

    if run_baseline:
        baseline_result = _build_logged_run_result(
            model=model,
            task=task,
            batch_id=batch_id,
            run_number=run_number,
            test_results=results.baseline_results,
            assertions_total=assertions_total,
            passed=results.baseline_success_rate > 0.5,
            run_type="baseline",
            skill_name=skill_name,
        )
        _persist_logged_run(create_run_folder(batch_folder, run_number), baseline_result)
        persisted_results.append(baseline_result)
        run_number += 1

    with_skill_result = _build_logged_run_result(
        model=model,
        task=task,
        batch_id=batch_id,
        run_number=run_number,
        test_results=results.with_skill_results,
        assertions_total=assertions_total,
        passed=with_skill_passed,
        run_type="with_skill",
        skill_name=skill_name,
    )
    _persist_logged_run(create_run_folder(batch_folder, run_number), with_skill_result)
    persisted_results.append(with_skill_result)
    return persisted_results


def _load_test_cases_from_payload(data: object) -> list[TestCase]:
    """Normalize test case JSON payloads into ``TestCase`` objects."""
    payload: object
    if isinstance(data, dict):
        mapping = cast("dict[object, object]", data)
        payload = mapping.get("cases", data)
    else:
        payload = data
    if not isinstance(payload, list):
        raise click.ClickException("Test payload must be a list or an object with `cases`.")
    return [TestCase.model_validate(test_case) for test_case in payload]


async def _load_test_cases(
    *,
    config: Config,
    skill_record: SkillRecord,
    tests_path: str | None,
    test_gen_model: str,
) -> tuple[list[TestCase], str]:
    """Load explicit, persisted, or generated test cases for a skill."""
    if tests_path:
        with open(tests_path, encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return _load_test_cases_from_payload(data), f"tests file: {tests_path}"

    if skill_record.state.tests:
        return skill_record.state.tests, "skill_meta.json"

    async with _fast_agent_context(config) as agent:
        console.print("Generating test cases from skill...", style="dim")
        await _set_agent_model(agent.test_gen, test_gen_model)
        test_cases = await generate_tests(
            skill_record.skill.description,
            generator=agent.test_gen,
            model=test_gen_model,
        )
    return test_cases, "generated"


def _count_invalid_expected_cases(test_cases: list[TestCase]) -> int:
    """Count generated or loaded tests missing enough expected strings."""
    invalid_expected = 0
    for test_case in test_cases:
        expected_values = [value.strip() for value in test_case.expected.contains if value.strip()]
        if len(expected_values) < 2:
            invalid_expected += 1
    return invalid_expected


def _load_trace_context(trace_path: Path) -> str:
    """Load a trace file into a prompt-sized context snippet."""
    trace_content = trace_path.read_text(encoding="utf-8")
    if trace_path.suffix.lower() != ".json":
        return trace_content[:4000]

    try:
        trace_data = json.loads(trace_content)
    except json.JSONDecodeError:
        return trace_content[:4000]
    return json.dumps(trace_data, indent=2)[:4000]


async def _create_generate_skill_record(
    *,
    task: str,
    examples: list[str] | None,
    from_skill: str | None,
    from_trace: str | None,
    agent: FastAgentSession,
    skill_gen_model: str,
) -> tuple[SkillRecord, str]:
    """Create or improve the skill record used by ``generate``."""
    await _set_agent_model(agent.skill_gen, skill_gen_model)

    if from_trace:
        trace_path = Path(from_trace)
        console.print(f"Generating skill from trace: {from_trace}", style="dim")
        task_with_trace = (
            f"{task}\n\nBased on this agent trace:\n\n{_load_trace_context(trace_path)}"
        )
        console.print(f"Generating skill with {skill_gen_model}...", style="dim")
        return (
            await generate_skill(
                task=task_with_trace,
                examples=examples,
                generator=agent.skill_gen,
                model=skill_gen_model,
            ),
            task_with_trace,
        )

    if from_skill:
        existing_skill = SkillRecord.load(Path(from_skill))
        console.print(
            f"Improving [bold]{existing_skill.skill.name}[/bold] with {skill_gen_model}...",
            style="dim",
        )
        return (
            await improve_skill(
                existing_skill,
                instructions=task,
                generator=agent.skill_gen,
                model=skill_gen_model,
            ),
            task,
        )

    console.print(f"Generating skill with {skill_gen_model}...", style="dim")
    return (
        await generate_skill(
            task=task,
            examples=examples,
            generator=agent.skill_gen,
            model=skill_gen_model,
        ),
        task,
    )


async def _submit_remote_eval_jobs(
    *,
    skill: Skill,
    test_cases: list[TestCase],
    model: str,
    jobs_config: JobsConfig,
    fastagent_config_path: Path,
    cards_path: Path,
    artifact_root: Path,
    run_baseline: bool,
) -> list[str]:
    """Submit remote fast-agent requests for an evaluation batch."""
    remote_executor = RemoteFastAgentExecutor(
        jobs_config=jobs_config,
        progress_callback=_print_eval_progress,
    )
    requests = build_eval_requests(
        skill=skill,
        test_cases=test_cases,
        model=model,
        fastagent_config_path=fastagent_config_path,
        cards_source_dir=cards_path,
        artifact_root=artifact_root,
        run_baseline=run_baseline,
    )
    job_refs: list[str] = []
    for pending_request in requests:
        submission = await remote_executor.submit(pending_request.request)
        job_refs.append(submission.job_id)
    return job_refs


async def _submit_generate_jobs_eval(
    *,
    skill: Skill,
    test_cases: list[TestCase],
    model: str,
    jobs_config: JobsConfig,
    config: Config,
    cards_path: Path,
    batch_folder: Path,
) -> list[str]:
    """Submit generate-time remote fast-agent requests without waiting for results."""
    return await _submit_remote_eval_jobs(
        skill=skill,
        test_cases=test_cases,
        model=model,
        jobs_config=jobs_config,
        fastagent_config_path=config.effective_fastagent_config,
        cards_path=cards_path,
        artifact_root=batch_folder / "remote_downloads" / "attempt_1",
        run_baseline=True,
    )


async def _run_generate_refinement_loop(
    *,
    skill_record: SkillRecord,
    task: str,
    test_cases: list[TestCase],
    executor: Executor,
    config: Config,
    cards_path: Path,
    batch_id: str,
    batch_folder: Path,
    skill_gen_model: str,
    log_runs: bool,
    max_parallel: int,
    agent: FastAgentSession,
) -> tuple[SkillRecord, EvalResults | None, list[RunResult]]:
    """Run generate-time eval/refinement attempts on the main model."""
    run_results: list[RunResult] = []
    prev_success_rate = 0.0
    results: EvalResults | None = None
    attempts = max(1, config.max_refine_attempts)

    for attempt in range(attempts):
        attempt_number = attempt + 1
        console.print(f"Evaluating on {skill_gen_model}... (attempt {attempt_number})", style="dim")
        console.print("[dim]Starting evaluation run...[/dim]")

        results = await evaluate_skill(
            skill_record.skill,
            test_cases=test_cases,
            executor=executor,
            model=skill_gen_model,
            fastagent_config_path=config.effective_fastagent_config,
            cards_source_dir=cards_path,
            artifact_root=batch_folder / f"attempt_{attempt_number}",
            show_baseline_progress=False,
            max_parallel=max_parallel,
            progress_callback=_print_eval_progress,
        )

        if log_runs:
            run_results.extend(
                _persist_comparison_run_results(
                    batch_folder=batch_folder,
                    model=skill_gen_model,
                    task=task,
                    batch_id=batch_id,
                    first_run_number=attempt * 2 + 1,
                    results=results,
                    assertions_total=len(test_cases),
                    run_baseline=True,
                    with_skill_passed=results.is_beneficial,
                    skill_name=skill_record.skill.name,
                )
            )

        lift = results.skill_lift
        lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"

        if results.is_beneficial:
            console.print(
                f"  {results.baseline_success_rate:.0%} -> "
                f"{results.with_skill_success_rate:.0%} ({lift_str}) [green]OK[/green]"
            )
            break

        console.print(
            f"  {results.baseline_success_rate:.0%} -> "
            f"{results.with_skill_success_rate:.0%} ({lift_str}) not good enough"
        )

        if abs(results.with_skill_success_rate - prev_success_rate) < 0.05:
            console.print("  [yellow]Plateaued, stopping[/yellow]")
            break

        prev_success_rate = results.with_skill_success_rate
        if attempt >= attempts - 1:
            continue

        console.print("Refining...", style="dim")
        failures = get_failure_descriptions(results)
        await _set_agent_model(agent.skill_gen, skill_gen_model)
        skill_record = await refine_skill(
            skill_record,
            failures,
            generator=agent.skill_gen,
            model=skill_gen_model,
        )

    return skill_record, results, run_results


async def _run_generate_extra_eval(
    *,
    skill_record: SkillRecord,
    task: str,
    test_cases: list[TestCase],
    executor: Executor,
    config: Config,
    cards_path: Path,
    batch_id: str,
    batch_folder: Path,
    model: str,
    log_runs: bool,
    max_parallel: int,
    first_run_number: int,
) -> tuple[EvalResults, list[RunResult]]:
    """Run the optional cross-model eval pass for ``generate``."""
    console.print(f"Evaluating on {model}...", style="dim")
    results = await evaluate_skill(
        skill_record.skill,
        test_cases,
        executor=executor,
        model=model,
        fastagent_config_path=config.effective_fastagent_config,
        cards_source_dir=cards_path,
        artifact_root=batch_folder / f"eval_{model}",
        show_baseline_progress=False,
        max_parallel=max_parallel,
        progress_callback=_print_eval_progress,
    )

    run_results: list[RunResult] = []
    if log_runs:
        run_results = _persist_comparison_run_results(
            batch_folder=batch_folder,
            model=model,
            task=task,
            batch_id=batch_id,
            first_run_number=first_run_number,
            results=results,
            assertions_total=len(test_cases),
            run_baseline=True,
            with_skill_passed=results.is_beneficial,
            skill_name=skill_record.skill.name,
        )

    lift = results.skill_lift
    lift_str = f"+{lift:.0%}" if lift > 0 else f"{lift:.0%}"
    console.print(
        f"  {results.baseline_success_rate:.0%} -> "
        f"{results.with_skill_success_rate:.0%} ({lift_str})"
    )
    return results, run_results


async def _run_with_skill_benchmark(
    *,
    skill_record: SkillRecord,
    evaluation_models: list[str],
    num_runs: int,
    test_cases: list[TestCase],
    executor: Executor,
    config: Config,
    cards_path: Path,
    batch_id: str,
    batch_folder: Path,
    verbose: bool,
    log_runs: bool,
    max_parallel: int,
) -> tuple[dict[str, list[RunResult]], list[RunResult]]:
    """Run a with-skill-only benchmark matrix across models and runs."""
    skill = skill_record.skill
    model_results: dict[str, list[RunResult]] = {model: [] for model in evaluation_models}
    all_run_results: list[RunResult] = []

    for model in evaluation_models:
        console.print(f"[bold]{model}[/bold]")

        for run_num in range(1, num_runs + 1):
            run_folder = create_run_folder(batch_folder, len(all_run_results) + 1)
            results = await evaluate_skill(
                skill,
                test_cases,
                executor=executor,
                model=model,
                fastagent_config_path=config.effective_fastagent_config,
                cards_source_dir=cards_path,
                artifact_root=run_folder / "eval",
                run_baseline=False,
                max_parallel=max_parallel,
                progress_callback=_print_eval_progress if verbose else None,
            )
            run_result = _build_logged_run_result(
                model=model,
                task=skill.description,
                batch_id=batch_id,
                run_number=run_num,
                test_results=results.with_skill_results,
                assertions_total=len(test_cases),
                passed=results.with_skill_success_rate > 0.5,
                run_type="with_skill",
                skill_name=skill.name,
            )

            if log_runs:
                _persist_logged_run(run_folder, run_result)

            model_results[model].append(run_result)
            all_run_results.append(run_result)

            if verbose:
                status = "[green]PASS[/green]" if run_result.passed else "[red]FAIL[/red]"
                console.print(
                    f"  Run {run_num}: {status} "
                    f"({run_result.assertions_passed}/{run_result.assertions_total} "
                    "assertions passed)"
                )

        console.print()

    return model_results, all_run_results


def _print_benchmark_summary(model_results: dict[str, list[RunResult]]) -> None:
    """Render the standard per-model benchmark summary."""
    console.print("\n[bold]Summary[/bold]\n")
    for model, results in model_results.items():
        total_runs = len(results)
        passed_runs = sum(1 for result in results if result.passed)
        avg_tokens = (
            sum(result.stats.total_tokens for result in results) / total_runs if total_runs else 0
        )
        avg_turns = sum(result.stats.turns for result in results) / total_runs if total_runs else 0
        pass_rate = passed_runs / total_runs if total_runs else 0
        if pass_rate > 0.5:
            pass_rate_style = "green"
        elif pass_rate > 0:
            pass_rate_style = "yellow"
        else:
            pass_rate_style = "red"

        console.print(f"[bold]{model}[/bold]")
        console.print(
            "  Runs: "
            f"{total_runs} | Passed: {passed_runs} ([{pass_rate_style}]"
            f"{pass_rate:.0%}[/{pass_rate_style}])"
        )
        console.print(f"  Avg tokens: {avg_tokens:.0f} | Avg turns: {avg_turns:.1f}")
        console.print()


def _write_benchmark_summary(
    *,
    batch_folder: Path,
    batch_id: str,
    evaluation_models: list[str],
    task: str,
    all_run_results: list[RunResult],
) -> None:
    """Persist the standard benchmark batch summary."""
    summary = BatchSummary(
        batch_id=batch_id,
        model=", ".join(evaluation_models),
        task=task,
        total_runs=len(all_run_results),
        passed_runs=sum(1 for result in all_run_results if result.passed),
        results=all_run_results,
    )
    write_batch_summary(batch_folder, summary)


def _load_eval_results(runs_path: Path) -> list[EvalPlotResult]:
    """Load eval results from batch summaries or run folders."""
    results: list[EvalPlotResult] = []
    if not runs_path.exists():
        return results

    batch_folders = sorted(p for p in runs_path.iterdir() if p.is_dir())
    for batch_folder in batch_folders:
        run_results: list[RunResult] = []
        summary = load_batch_summary(batch_folder)
        if summary:
            run_results = summary.results
        else:
            for run_folder in sorted(batch_folder.glob("run_*")):
                if not run_folder.is_dir():
                    continue
                run_result = load_run_result(run_folder)
                if run_result:
                    run_results.append(run_result)

        if not run_results:
            continue

        grouped: dict[tuple[str, str], dict[str, list[RunResult]]] = {}
        for run in run_results:
            skill_name = run.skill_name or "unknown"
            key = (run.metadata.model, skill_name)
            grouped.setdefault(key, {"baseline": [], "with_skill": []})
            if run.run_type == "baseline":
                grouped[key]["baseline"].append(run)
            else:
                grouped[key]["with_skill"].append(run)

        for (model, skill_name), runs in grouped.items():
            baseline_runs = sorted(runs["baseline"], key=lambda r: r.metadata.run_number)
            for with_skill_run in runs["with_skill"]:
                baseline_run = _select_baseline_run(baseline_runs, with_skill_run)
                result: EvalPlotResult = {
                    "model": model,
                    "skill_name": skill_name,
                    "with_skill_rate": _success_rate(with_skill_run),
                    "with_skill_tokens": with_skill_run.stats.total_tokens,
                    "baseline_rate": _success_rate(baseline_run) if baseline_run else 0.0,
                    "baseline_tokens": baseline_run.stats.total_tokens if baseline_run else 0,
                    "has_baseline": baseline_run is not None,
                }
                results.append(result)

    return results


@click.group()
@click.version_option()
def main():
    """upskill - Generate and evaluate agent skills."""
    resolution = resolve_upskill_config_path()
    if resolution.path is None:
        console.print("[dim]Config source: defaults (no config file found)[/dim]")
        return

    message = f"[dim]Config source: {resolution.source} ({resolution.path})[/dim]"
    if not resolution.exists:
        message += " [yellow](file missing; using defaults until saved)[/yellow]"
    console.print(message)


@main.command()
@click.argument("task")
@click.option("-e", "--example", multiple=True, help="Input -> output example")
@click.option("--tool", help="Generate from MCP tool schema (path#tool_name)")
@click.option(
    "-f",
    "--from",
    "from_source",
    type=click.Path(exists=True),
    help="Improve existing skill (directory) or generate from trace (file)",
)
@click.option(
    "-m",
    "--model",
    help="Skill generation model for skill creation/refinement",
)
@click.option(
    "--test-gen-model",
    help="Override test generation model for this run",
)
@click.option("-o", "--output", type=click.Path(), help="Output directory for skill")
@click.option("--no-eval", is_flag=True, help="Skip eval and refinement")
@click.option("--eval-model", help="Optional extra cross-model eval pass after generation")
@_jobs_execution_options(
    executor_help="Execution backend for evaluation/refinement runs",
    runs_dir_help="Directory for run logs (default: ./runs)",
)
def generate(
    task: str,
    example: tuple[str, ...],
    tool: str | None,
    from_source: str | None,
    model: str | None,
    test_gen_model: str | None,
    output: str | None,
    no_eval: bool,
    eval_model: str | None,
    executor: ExecutorName,
    artifact_repo: str | None,
    wait: bool,
    jobs_timeout: str,
    jobs_flavor: str,
    jobs_secrets: str,
    jobs_namespace: str | None,
    max_parallel: int,
    runs_dir: str | None,
    log_runs: bool,
):
    """Generate a skill from a task description, or improve an existing skill.

    Examples:

        upskill generate "parse JSON Schema files"

        upskill generate "write git commits" --model sonnet

        upskill generate "write git commits" --model sonnet --test-gen-model opus

        upskill generate "handle API errors" --eval-model haiku

        upskill generate "validate forms" -o ./my-skills/validation

        # Improve an existing skill (auto-detects directory):

        upskill generate "add more error handling examples" --from ./skills/api-errors/

        upskill generate "make it more concise" -f ./skills/my-skill/ -o ./skills/my-skill-v2/

        # Generate from trace file (auto-detects file):

        upskill generate "extract patterns" --from trace.json

        # Skip evaluation (evaluate separately with upskill eval)

        upskill generate "parse YAML" --no-eval

        upskill generate "document code" --no-log-runs
    """
    del tool

    # Auto-detect if --from is a skill directory or trace file
    from_skill = None
    from_trace = None
    if from_source:
        source_path = Path(from_source)
        if source_path.is_dir():
            from_skill = from_source
        else:
            from_trace = from_source

    asyncio.run(
        _generate_async(
            task,
            list(example) if example else None,
            from_skill,
            from_trace,
            model,
            test_gen_model,
            output,
            no_eval,
            eval_model,
            executor,
            artifact_repo,
            wait,
            jobs_timeout,
            jobs_flavor,
            jobs_secrets,
            jobs_namespace,
            max_parallel,
            runs_dir,
            log_runs,
        )
    )


async def _generate_async(
    task: str,
    examples: list[str] | None,
    from_skill: str | None,
    from_trace: str | None,
    model: str | None,
    test_gen_model: str | None,
    output: str | None,
    no_eval: bool,
    eval_model: str | None,
    executor_name: ExecutorName,
    artifact_repo: str | None,
    wait: bool,
    jobs_timeout: str,
    jobs_flavor: str,
    jobs_secrets: str,
    jobs_namespace: str | None,
    max_parallel: int,
    runs_dir: str | None,
    log_runs: bool,
):
    """Async implementation of generate command."""
    config = Config.load()
    jobs_config = _require_jobs_config(
        executor_name=executor_name,
        artifact_repo=artifact_repo,
        wait=wait,
        jobs_timeout=jobs_timeout,
        jobs_flavor=jobs_flavor,
        jobs_secrets=jobs_secrets,
        jobs_namespace=jobs_namespace,
    )
    resolved = resolve_models(
        "generate",
        config=config,
        cli_model=model,
        cli_eval_model=eval_model,
        cli_test_gen_model=test_gen_model,
    )
    skill_gen_model = _require_resolved_model(
        resolved.skill_generation_model,
        field="skill_generation_model",
        command="generate",
    )
    test_gen_model = _require_resolved_model(
        resolved.test_generation_model,
        field="test_generation_model",
        command="generate",
    )
    extra_eval_model = resolved.extra_eval_model

    _print_model_plan("generate", resolved)

    # Setup artifact storage and optional run logging
    runs_path = Path(runs_dir) if runs_dir else config.runs_dir
    batch_id, batch_folder = create_batch_folder(runs_path)
    run_results: list[RunResult] = []
    console.print(f"Artifacts saved to: {batch_folder}", style="dim")
    if log_runs:
        console.print(f"Logging runs to: {batch_folder}", style="dim")

    async with _fast_agent_context(config) as agent:
        cards = resources.files("upskill").joinpath("agent_cards")
        with resources.as_file(cards) as cards_path:
            skill_record, eval_task = await _create_generate_skill_record(
                task=task,
                examples=examples,
                from_skill=from_skill,
                from_trace=from_trace,
                agent=agent,
                skill_gen_model=skill_gen_model,
            )

            console.print("Generating test cases...", style="dim")
            await _set_agent_model(agent.test_gen, test_gen_model)
            test_cases = await generate_tests(
                eval_task,
                generator=agent.test_gen,
                model=test_gen_model,
            )
            skill_record.state.tests = list(test_cases)

            if no_eval:
                _save_and_display(skill_record, output, config, artifact_path=batch_folder)
                return

            if executor_name == "jobs" and not wait:
                if jobs_config is None:
                    raise RuntimeError("Jobs config was not initialized.")
                job_refs = await _submit_generate_jobs_eval(
                    skill=skill_record.skill,
                    test_cases=test_cases,
                    model=skill_gen_model,
                    jobs_config=jobs_config,
                    config=config,
                    cards_path=cards_path,
                    batch_folder=batch_folder,
                )
                console.print(
                    "[yellow]Remote fast-agent requests submitted without --wait; "
                    "refinement is skipped for this run.[/yellow]"
                )
                console.print(f"Remote fast-agent job id(s): {', '.join(job_refs)}")
                _save_and_display(skill_record, output, config, artifact_path=batch_folder)
                return

            executor = _build_executor(
                executor_name,
                jobs_config=jobs_config,
                progress_callback=_print_eval_progress,
            )

            skill_record, results, run_results = await _run_generate_refinement_loop(
                skill_record=skill_record,
                task=eval_task,
                test_cases=test_cases,
                executor=executor,
                config=config,
                cards_path=cards_path,
                batch_id=batch_id,
                batch_folder=batch_folder,
                skill_gen_model=skill_gen_model,
                log_runs=log_runs,
                max_parallel=max_parallel,
                agent=agent,
            )

            # If eval_model specified, also eval on that model
            eval_results = None
            if extra_eval_model:
                eval_results, extra_run_results = await _run_generate_extra_eval(
                    skill_record=skill_record,
                    task=eval_task,
                    test_cases=test_cases,
                    executor=executor,
                    config=config,
                    cards_path=cards_path,
                    batch_id=batch_id,
                    batch_folder=batch_folder,
                    model=extra_eval_model,
                    log_runs=log_runs,
                    max_parallel=max_parallel,
                    first_run_number=len(run_results) + 1,
                )
                run_results.extend(extra_run_results)

            # Write batch summary
            if log_runs:
                summary = BatchSummary(
                    batch_id=batch_id,
                    model=skill_gen_model,
                    task=eval_task,
                    total_runs=len(run_results),
                    passed_runs=sum(1 for r in run_results if r.passed),
                    results=run_results,
                )
                write_batch_summary(batch_folder, summary)

    if not no_eval:
        if results:
            skill_record.state.metadata.test_pass_rate = results.with_skill_success_rate
        else:
            console.print(
                "[yellow]No evaluation results available; skipping report output.[/yellow]"
            )

        _save_and_display(
            skill_record,
            output,
            config,
            results,
            eval_results,
            skill_gen_model,
            extra_eval_model,
            batch_folder,
        )


def _save_and_display(
    skill_record: SkillRecord,
    output: str | None,
    config: Config,
    results: EvalResults | None = None,
    eval_results: EvalResults | None = None,
    skill_gen_model: str | None = None,
    eval_model: str | None = None,
    artifact_path: Path | None = None,
):
    """Save skill and display summary."""
    skill = skill_record.skill
    output_path = Path(output) if output else config.skills_dir / skill.name

    skill_record.save(output_path)

    console.print("[dim]Rendering report output...[/dim]")

    console.print()
    console.print(f"  [bold]{skill.name}[/bold]")
    console.print(f"  {skill.description}")
    console.print()

    skill_tokens = len(skill.body.split()) * 1.3
    console.print(f"  SKILL.md              ~{int(skill_tokens)} tokens")
    for name in skill.references:
        ref_tokens = len(skill.references[name].split()) * 1.3
        console.print(f"  references/{name}  ~{int(ref_tokens)} tokens")
    for name in skill.scripts:
        console.print(f"  scripts/{name}     (exec only)")

    # Show results with horizontal bars
    if results and eval_results:
        # Multiple models - show each with bars
        console.print()
        model_rows = [
            (skill_gen_model or "skill-gen", results),
            (eval_model or "eval", eval_results),
        ]
        for model_name, r in model_rows:
            console.print(f"  [bold]{model_name}[/bold]")
            baseline_bar = _render_bar(r.baseline_success_rate)
            with_skill_bar = _render_bar(r.with_skill_success_rate)
            lift = r.skill_lift
            lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
            lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"

            console.print(f"    baseline   {baseline_bar}  {r.baseline_success_rate:>5.0%}")
            console.print(
                f"    with skill {with_skill_bar}  {r.with_skill_success_rate:>5.0%}  "
                f"[{lift_style}]({lift_str})[/{lift_style}]"
            )
            console.print()

    elif results:
        # Single model
        console.print()
        baseline_bar = _render_bar(results.baseline_success_rate)
        with_skill_bar = _render_bar(results.with_skill_success_rate)
        lift = results.skill_lift
        lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
        lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"

        console.print(f"  baseline   {baseline_bar}  {results.baseline_success_rate:>5.0%}")
        console.print(
            f"  with skill {with_skill_bar}  {results.with_skill_success_rate:>5.0%}  "
            f"[{lift_style}]({lift_str})[/{lift_style}]"
        )

        if results.baseline_total_tokens > 0:
            savings = results.token_savings
            savings_str = f"-{savings:.0%}" if savings >= 0 else f"+{-savings:.0%}"
            savings_style = "green" if savings > 0 else "red" if savings < 0 else "dim"
            console.print()
            console.print(
                f"  tokens: {results.baseline_total_tokens} → {results.with_skill_total_tokens}  "
                f"[{savings_style}]({savings_str})[/{savings_style}]"
            )

    console.print()
    console.print(f"Saved to {output_path}")
    if artifact_path is not None:
        console.print(f"Artifacts saved to {artifact_path}")


@main.command("eval")
@click.argument("skill_path", type=click.Path(exists=True))
@click.option("-t", "--tests", type=click.Path(exists=True), help="Test cases JSON file")
@click.option(
    "-m",
    "--model",
    "models",
    multiple=True,
    help="Evaluation model(s) to run tests on (repeatable)",
)
@click.option(
    "--test-gen-model",
    help="Override test generation model when tests must be generated",
)
@click.option("--runs", "num_runs", type=int, default=1, help="Number of runs per model")
@click.option(
    "--no-baseline",
    is_flag=True,
    help="Skip baseline comparison in simple eval mode (ignored in benchmark mode)",
)
@click.option("-v", "--verbose", is_flag=True, help="Show per-test results")
@_jobs_execution_options(
    executor_help="Execution backend for evaluation runs",
    runs_dir_help="Directory for run logs",
)
def eval_cmd(
    skill_path: str,
    tests: str | None,
    models: tuple[str, ...],
    test_gen_model: str | None,
    num_runs: int,
    no_baseline: bool,
    verbose: bool,
    executor: ExecutorName,
    artifact_repo: str | None,
    wait: bool,
    jobs_timeout: str,
    jobs_flavor: str,
    jobs_secrets: str,
    jobs_namespace: str | None,
    max_parallel: int,
    log_runs: bool,
    runs_dir: str | None,
):
    """Evaluate a skill.

    Uses simple eval mode for one model with ``--runs 1``.
    Enters benchmark mode when using multiple ``-m`` values or ``--runs > 1``.

    Examples:

        upskill eval ./skills/my-skill/

        upskill eval ./skills/my-skill/ --tests ./tests.json -v

        upskill eval ./skills/my-skill/ -m haiku

        upskill eval ./skills/my-skill/ -m haiku --test-gen-model opus

        upskill eval ./skills/my-skill/ -m haiku -m sonnet

        upskill eval ./skills/my-skill/ -m haiku --runs 5

        # Evaluate local models configured in fast-agent

        upskill eval ./skills/my-skill/ -m generic.llama3.2:latest

        upskill eval ./skills/my-skill/ --no-log-runs
    """
    asyncio.run(
        _eval_async(
            skill_path,
            tests,
            list(models) if models else None,
            test_gen_model,
            num_runs,
            no_baseline,
            verbose,
            executor,
            artifact_repo,
            wait,
            jobs_timeout,
            jobs_flavor,
            jobs_secrets,
            jobs_namespace,
            max_parallel,
            log_runs,
            runs_dir,
        )
    )


async def _eval_async(  # noqa: C901
    skill_path: str,
    tests: str | None,
    models: list[str] | None,
    test_gen_model: str | None,
    num_runs: int,
    no_baseline: bool,
    verbose: bool,
    executor_name: ExecutorName,
    artifact_repo: str | None,
    wait: bool,
    jobs_timeout: str,
    jobs_flavor: str,
    jobs_secrets: str,
    jobs_namespace: str | None,
    max_parallel: int,
    log_runs: bool,
    runs_dir: str | None,
):
    """Async implementation of eval command."""
    config = Config.load()
    jobs_config = _require_jobs_config(
        executor_name=executor_name,
        artifact_repo=artifact_repo,
        wait=wait,
        jobs_timeout=jobs_timeout,
        jobs_flavor=jobs_flavor,
        jobs_secrets=jobs_secrets,
        jobs_namespace=jobs_namespace,
    )
    executor = None
    if executor_name == "local" or wait:
        executor = _build_executor(
            executor_name,
            jobs_config=jobs_config,
            progress_callback=_print_eval_progress,
        )
    resolved = resolve_models(
        "eval",
        config=config,
        cli_models=models,
        cli_test_gen_model=test_gen_model,
        num_runs=num_runs,
        no_baseline=no_baseline,
    )
    evaluation_models = _require_resolved_models(
        resolved.evaluation_models,
        field="evaluation_models",
        command="eval",
    )
    test_gen_model = _require_resolved_model(
        resolved.test_generation_model,
        field="test_generation_model",
        command="eval",
    )

    _print_model_plan("eval", resolved, runs=num_runs)
    if resolved.is_benchmark_mode and no_baseline:
        console.print(
            "[dim]Note: --no-baseline is redundant in benchmark mode and is ignored.[/dim]"
        )

    skill_dir = Path(skill_path)

    try:
        skill_record = SkillRecord.load(skill_dir)
    except FileNotFoundError:
        console.print(f"[red]No SKILL.md found in {skill_dir}[/red]")
        sys.exit(1)
    skill = skill_record.skill

    test_cases, test_source = await _load_test_cases(
        config=config,
        skill_record=skill_record,
        tests_path=tests,
        test_gen_model=test_gen_model,
    )

    invalid_expected = _count_invalid_expected_cases(test_cases)
    console.print(f"[dim]Loaded {len(test_cases)} test case(s) from {test_source}[/dim]")
    if invalid_expected:
        console.print(f"[yellow]{invalid_expected} test case(s) missing expected strings[/yellow]")

    runs_path = Path(runs_dir) if runs_dir else config.runs_dir
    batch_id, batch_folder = create_batch_folder(runs_path)
    console.print(f"Artifacts saved to: {batch_folder}", style="dim")
    if log_runs:
        console.print(f"Logging to: {batch_folder}", style="dim")

    if executor_name == "jobs" and not wait:
        if jobs_config is None:
            raise RuntimeError("Jobs config was not initialized.")
        cards = resources.files("upskill").joinpath("agent_cards")
        with resources.as_file(cards) as cards_path:
            if resolved.is_benchmark_mode:
                submitted_job_refs: list[str] = []

                for model in evaluation_models:
                    console.print(f"[bold]{model}[/bold]")
                    for run_num in range(1, num_runs + 1):
                        job_refs = await _submit_remote_eval_jobs(
                            skill=skill,
                            test_cases=test_cases,
                            model=model,
                            jobs_config=jobs_config,
                            fastagent_config_path=config.effective_fastagent_config,
                            cards_path=cards_path,
                            artifact_root=batch_folder
                            / "remote_downloads"
                            / model
                            / f"run_{run_num}",
                            run_baseline=False,
                        )
                        submitted_job_refs.extend(job_refs)
                        console.print(f"Remote fast-agent job id(s): {', '.join(job_refs)}")
                console.print(
                    f"Submitted remote fast-agent job id(s): {', '.join(submitted_job_refs)}"
                )
                return

            job_refs = await _submit_remote_eval_jobs(
                skill=skill,
                test_cases=test_cases,
                model=evaluation_models[0],
                jobs_config=jobs_config,
                fastagent_config_path=config.effective_fastagent_config,
                cards_path=cards_path,
                artifact_root=batch_folder / "remote_downloads",
                run_baseline=resolved.run_baseline,
            )
        console.print(f"Remote fast-agent job id(s): {', '.join(job_refs)}")
        return

    if executor is None:
        raise RuntimeError("Local executor was not initialized.")

    cards = resources.files("upskill").joinpath("agent_cards")
    with resources.as_file(cards) as cards_path:
        if resolved.is_benchmark_mode:
            console.print(
                f"\nEvaluating [bold]{skill.name}[/bold] across {len(evaluation_models)} model(s)"
            )
            console.print(f"  {len(test_cases)} test case(s), {num_runs} run(s) per model\n")
            model_results, all_run_results = await _run_with_skill_benchmark(
                skill_record=skill_record,
                evaluation_models=evaluation_models,
                num_runs=num_runs,
                test_cases=test_cases,
                executor=executor,
                config=config,
                cards_path=cards_path,
                batch_id=batch_id,
                batch_folder=batch_folder,
                verbose=verbose,
                log_runs=log_runs,
                max_parallel=max_parallel,
            )
            _print_benchmark_summary(model_results)
            if log_runs:
                _write_benchmark_summary(
                    batch_folder=batch_folder,
                    batch_id=batch_id,
                    evaluation_models=evaluation_models,
                    task=skill.description,
                    all_run_results=all_run_results,
                )

        else:
            # Simple eval mode: single model, single run
            model = evaluation_models[0]
            console.print(f"Running {len(test_cases)} test cases...", style="dim")

            results = await evaluate_skill(
                skill,
                test_cases,
                executor=executor,
                model=model,
                fastagent_config_path=config.effective_fastagent_config,
                cards_source_dir=cards_path,
                artifact_root=batch_folder / "eval",
                run_baseline=resolved.run_baseline,
                show_baseline_progress=verbose,
                max_parallel=max_parallel,
                progress_callback=_print_eval_progress,
            )

            # Log results (both baseline and with-skill)
            run_results: list[RunResult] = []
            if log_runs:
                run_results = _persist_comparison_run_results(
                    batch_folder=batch_folder,
                    model=model,
                    task=skill.description,
                    batch_id=batch_id,
                    first_run_number=1,
                    results=results,
                    assertions_total=len(test_cases),
                    run_baseline=resolved.run_baseline,
                    with_skill_passed=(
                        results.is_beneficial
                        if resolved.run_baseline
                        else results.with_skill_success_rate > 0.5
                    ),
                    skill_name=skill.name,
                )

                # Write batch summary
                summary = BatchSummary(
                    batch_id=batch_id,
                    model=model,
                    task=skill.description,
                    total_runs=len(run_results),
                    passed_runs=sum(1 for r in run_results if r.passed),
                    results=run_results,
                )
                write_batch_summary(batch_folder, summary)

            if verbose and resolved.run_baseline:
                console.print()
                for i, (with_r, base_r) in enumerate(
                    zip(results.with_skill_results, results.baseline_results, strict=True),
                    1,
                ):
                    base_icon = "[green]OK[/green]" if base_r.success else "[red]FAIL[/red]"
                    skill_icon = "[green]OK[/green]" if with_r.success else "[red]FAIL[/red]"
                    input_preview = with_r.test_case.input[:40]
                    console.print(f"  {i}. {input_preview}  {base_icon} base  {skill_icon} skill")
                console.print()

            # Display results with horizontal bars
            console.print()
            if resolved.run_baseline:
                baseline_rate = results.baseline_success_rate
                with_skill_rate = results.with_skill_success_rate
                lift = results.skill_lift

                baseline_bar = _render_bar(baseline_rate)
                with_skill_bar = _render_bar(with_skill_rate)

                lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
                lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"

                console.print(f"  baseline   {baseline_bar}  {baseline_rate:>5.0%}")
                console.print(
                    f"  with skill {with_skill_bar}  {with_skill_rate:>5.0%}  "
                    f"[{lift_style}]({lift_str})[/{lift_style}]"
                )

                # Token comparison
                if results.baseline_total_tokens > 0:
                    savings = results.token_savings
                    savings_str = f"-{savings:.0%}" if savings >= 0 else f"+{-savings:.0%}"
                    savings_style = "green" if savings > 0 else "red" if savings < 0 else "dim"
                    console.print()
                    token_line = (
                        f"  tokens: {results.baseline_total_tokens} → "
                        f"{results.with_skill_total_tokens}  "
                        f"[{savings_style}]({savings_str})[/{savings_style}]"
                    )
                    console.print(token_line)
            else:
                with_skill_rate = results.with_skill_success_rate
                with_skill_bar = _render_bar(with_skill_rate)
                console.print(f"  with skill {with_skill_bar}  {with_skill_rate:>5.0%}")
                console.print(f"  tokens: {results.with_skill_total_tokens}")

            console.print(f"\nArtifacts saved to: {batch_folder}")
            if resolved.run_baseline:
                if results.is_beneficial:
                    console.print("\n[green]Recommendation: keep skill[/green]")
                else:
                    console.print("\n[yellow]Recommendation: skill may not be beneficial[/yellow]")


@main.command("list")
@click.option("-d", "--dir", "skills_dir", type=click.Path(), help="Skills directory to list")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed skill structure")
def list_cmd(skills_dir: str | None, verbose: bool):
    """List generated skills."""
    config = Config.load()
    path = Path(skills_dir) if skills_dir else config.skills_dir

    if not path.exists():
        console.print(f"No skills directory found at {path}")
        return

    skills = [d for d in path.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]

    if not skills:
        console.print(f"No skills found in {path}")
        return

    # Build tree view
    tree = Tree(f"[bold]Skills in {path}[/bold]")
    total_tokens = 0

    for skill_dir in sorted(skills):
        skill_md = skill_dir / "SKILL.md"
        content = skill_md.read_text()
        lines = content.split("\n")
        name = skill_dir.name

        # Estimate tokens
        skill_tokens = int(len(content.split()) * 1.3)
        total_tokens += skill_tokens

        skill_branch = tree.add(f"[bold]{name}[/bold]")

        if verbose:
            # Add SKILL.md with token count
            skill_branch.add(f"SKILL.md [dim](~{skill_tokens} tokens)[/dim]")

            # Check for references directory
            refs_dir = skill_dir / "references"
            if refs_dir.exists() and refs_dir.is_dir():
                refs_branch = skill_branch.add("references/")
                for ref_file in sorted(refs_dir.iterdir()):
                    if ref_file.is_file():
                        ref_tokens = int(len(ref_file.read_text().split()) * 1.3)
                        total_tokens += ref_tokens
                        refs_branch.add(f"{ref_file.name} [dim](~{ref_tokens} tokens)[/dim]")

            # Check for scripts directory
            scripts_dir = skill_dir / "scripts"
            if scripts_dir.exists() and scripts_dir.is_dir():
                scripts_branch = skill_branch.add("scripts/")
                for script_file in sorted(scripts_dir.iterdir()):
                    if script_file.is_file():
                        scripts_branch.add(f"{script_file.name}")

            # Check for tests.json
            tests_file = skill_dir / "tests.json"
            if tests_file.exists():
                with open(tests_file, encoding="utf-8") as f:
                    tests_data = json.load(f)
                num_cases = len(tests_data.get("cases", tests_data))
                skill_branch.add(f"tests.json [dim]({num_cases} cases)[/dim]")
        else:
            # Simple view: just show description
            description = lines[2] if len(lines) > 2 else ""
            if description:
                skill_branch.add(f"[dim]{description[:60]}...[/dim]")

    console.print()
    console.print(tree)
    console.print()
    console.print(f"[dim]{len(skills)} skills, ~{total_tokens:,} total tokens[/dim]")


@main.command("benchmark")
@click.argument("skill_path", type=click.Path(exists=True))
@click.option(
    "-m",
    "--model",
    "models",
    multiple=True,
    required=True,
    help="Evaluation model(s) to benchmark (repeatable)",
)
@click.option("--runs", "num_runs", type=int, default=3, help="Runs per model (default: 3)")
@click.option("-t", "--tests", type=click.Path(exists=True), help="Test cases JSON file")
@click.option(
    "--test-gen-model",
    help="Override test generation model when tests must be generated",
)
@click.option(
    "--executor",
    type=click.Choice(["local", "jobs"]),
    default="local",
    show_default=True,
    help="Execution backend for benchmark runs",
)
@click.option("--artifact-repo", help="Dataset repo for remote fast-agent job artifacts")
@click.option(
    "--wait/--no-wait", default=True, help="Wait for remote fast-agent jobs and download results"
)
@click.option(
    "--jobs-timeout",
    default="2h",
    show_default=True,
    help="HF Jobs timeout for remote fast-agent runs",
)
@click.option(
    "--jobs-flavor",
    default="cpu-basic",
    show_default=True,
    help="HF Jobs hardware flavor for remote fast-agent runs",
)
@click.option(
    "--jobs-secrets",
    default="HF_TOKEN",
    show_default=True,
    help="Comma-separated HF Job secrets to forward",
)
@click.option("--jobs-namespace", help="Optional Hugging Face Jobs namespace")
@click.option("-o", "--output", type=click.Path(), help="Output directory for results")
@click.option("-v", "--verbose", is_flag=True, help="Show per-run details")
@click.option(
    "--max-parallel",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="Maximum concurrent evaluation executions per phase",
)
def benchmark_cmd(
    skill_path: str,
    models: tuple[str, ...],
    num_runs: int,
    tests: str | None,
    test_gen_model: str | None,
    executor: ExecutorName,
    artifact_repo: str | None,
    wait: bool,
    jobs_timeout: str,
    jobs_flavor: str,
    jobs_secrets: str,
    jobs_namespace: str | None,
    output: str | None,
    verbose: bool,
    max_parallel: int,
):
    """Benchmark a skill across multiple models.

    Runs the skill's test cases multiple times per model and reports
    pass rates and assertion statistics. MCP servers from fastagent.config.yaml
    are automatically enabled.

    Examples:

        upskill benchmark ./skills/hf-eval-extraction/ -m haiku -m sonnet

        upskill benchmark ./skills/hf-eval-extraction/ -m haiku -m sonnet --test-gen-model opus

        upskill benchmark ./skills/my-skill/ -m gpt-4o -m claude-sonnet --runs 5

        upskill benchmark ./skills/my-skill/ -m haiku -t ./custom_tests.json -v
    """
    asyncio.run(
        _benchmark_async(
            skill_path,
            list(models),
            test_gen_model,
            num_runs,
            tests,
            executor,
            artifact_repo,
            wait,
            jobs_timeout,
            jobs_flavor,
            jobs_secrets,
            jobs_namespace,
            output,
            verbose,
            max_parallel,
        )
    )


async def _benchmark_async(
    skill_path: str,
    models: list[str],
    test_gen_model: str | None,
    num_runs: int,
    tests_path: str | None,
    executor_name: ExecutorName,
    artifact_repo: str | None,
    wait: bool,
    jobs_timeout: str,
    jobs_flavor: str,
    jobs_secrets: str,
    jobs_namespace: str | None,
    output_dir: str | None,
    verbose: bool,
    max_parallel: int,
):
    """Async implementation of benchmark command."""
    config = Config.load()
    jobs_config = _require_jobs_config(
        executor_name=executor_name,
        artifact_repo=artifact_repo,
        wait=wait,
        jobs_timeout=jobs_timeout,
        jobs_flavor=jobs_flavor,
        jobs_secrets=jobs_secrets,
        jobs_namespace=jobs_namespace,
    )
    if executor_name == "jobs" and not wait:
        raise click.ClickException(
            "`benchmark --executor jobs` currently requires `--wait` to assemble results from "
            "downloaded fast-agent artifacts."
        )
    executor = _build_executor(
        executor_name,
        jobs_config=jobs_config,
        progress_callback=_print_eval_progress,
    )
    resolved = resolve_models(
        "benchmark",
        config=config,
        cli_models=models,
        cli_test_gen_model=test_gen_model,
        num_runs=num_runs,
    )
    evaluation_models = _require_resolved_models(
        resolved.evaluation_models,
        field="evaluation_models",
        command="benchmark",
    )
    test_gen_model = _require_resolved_model(
        resolved.test_generation_model,
        field="test_generation_model",
        command="benchmark",
    )

    _print_model_plan("benchmark", resolved, runs=num_runs)

    skill_record = SkillRecord.load(Path(skill_path))
    skill = skill_record.skill

    cards = resources.files("upskill").joinpath("agent_cards")
    with resources.as_file(cards) as cards_path:
        test_cases, _ = await _load_test_cases(
            config=config,
            skill_record=skill_record,
            tests_path=tests_path,
            test_gen_model=test_gen_model,
        )

        # Setup output directory
        out_path = Path(output_dir) if output_dir else config.runs_dir

        batch_id, batch_folder = create_batch_folder(out_path)
        console.print(f"Results will be saved to: {batch_folder}", style="dim")

        console.print(
            f"\nBenchmarking [bold]{skill.name}[/bold] across {len(evaluation_models)} model(s)"
        )
        console.print(f"  {len(test_cases)} test case(s), {num_runs} run(s) per model\n")
        model_results, all_run_results = await _run_with_skill_benchmark(
            skill_record=skill_record,
            evaluation_models=evaluation_models,
            num_runs=num_runs,
            test_cases=test_cases,
            executor=executor,
            config=config,
            cards_path=cards_path,
            batch_id=batch_id,
            batch_folder=batch_folder,
            verbose=verbose,
            log_runs=True,
            max_parallel=max_parallel,
        )
        _print_benchmark_summary(model_results)
        _write_benchmark_summary(
            batch_folder=batch_folder,
            batch_id=batch_id,
            evaluation_models=evaluation_models,
            task=skill.description,
            all_run_results=all_run_results,
        )


@main.command("runs")
@click.option("-d", "--dir", "runs_dir", type=click.Path(exists=True), help="Runs directory")
@click.option("-s", "--skill", "skills", multiple=True, help="Filter by skill name(s)")
@click.option(
    "-m",
    "--model",
    "models",
    multiple=True,
    help="Filter historical run data by model(s)",
)
@click.option("--csv", "csv_output", type=click.Path(), help="Export to CSV file")
@click.option(
    "--metric",
    type=click.Choice(["success", "tokens"]),
    default="success",
    help="Metric to display",
)
def runs_cmd(
    runs_dir: str | None,
    skills: tuple[str, ...],
    models: tuple[str, ...],
    csv_output: str | None,
    metric: str,
):
    """Show evaluation history with visual comparison.

    By default shows a plot of baseline vs with-skill performance.
    Use --csv to export results to a CSV file.

    Examples:

        upskill runs

        upskill runs -s my-skill -m haiku -m sonnet

        upskill runs --csv results.csv

        upskill runs --metric tokens
    """
    config = Config.load()
    runs_path = Path(runs_dir) if runs_dir else config.runs_dir

    if not runs_path.exists():
        console.print(f"[red]No runs directory found at {runs_path}[/red]")
        sys.exit(1)

    # If --csv is specified, export to CSV
    if csv_output:
        try:
            output_path = summarize_runs_to_csv(runs_path, Path(csv_output))
            console.print(f"Summary written to {output_path}")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)
        return

    # Otherwise, show plot
    all_results = _load_eval_results(runs_path)

    if not all_results:
        console.print("[yellow]No eval results with baseline comparisons found.[/yellow]")
        console.print("Run 'upskill eval <skill>' to generate comparable results.")
        console.print("\n[dim]--csv results.csv to export raw data[/dim]")
        sys.exit(0)

    # Filter by skills and models
    if skills:
        all_results = [r for r in all_results if r["skill_name"] in skills]
    if models:
        all_results = [r for r in all_results if r["model"] in models]

    if not all_results:
        console.print("[yellow]No results match the specified filters.[/yellow]")
        sys.exit(0)

    # Aggregate by model and skill (take most recent / highest)
    aggregated: dict[tuple[str, str], EvalPlotResult] = {}
    for r in all_results:
        key = (r["model"], r["skill_name"])
        if key not in aggregated or r["with_skill_rate"] > aggregated[key]["with_skill_rate"]:
            aggregated[key] = r

    results_list: list[EvalPlotResult] = list(aggregated.values())

    # Determine display mode
    unique_skills = set(r["skill_name"] for r in results_list)
    unique_models = set(r["model"] for r in results_list)

    console.print()

    if len(unique_skills) == 1 and len(unique_models) >= 1:
        # Single skill, multiple models - use Panel
        skill_name = next(iter(unique_skills))

        # Build content for panel
        content_lines = []
        for r in sorted(results_list, key=lambda x: x["model"]):
            content_lines.append(_format_comparison_bars(r, metric))

        panel_content = "\n".join(content_lines)
        panel_title = f"Evaluation History: {skill_name}"
        console.print(Panel(panel_content, title=panel_title, border_style="blue"))

    elif len(unique_models) == 1 and len(unique_skills) >= 1:
        # Single model, multiple skills - use Panel
        model_name = next(iter(unique_models))

        content_lines = []
        for r in sorted(results_list, key=lambda x: x["skill_name"]):
            content_lines.append(_format_comparison_bars(r, metric, label_field="skill_name"))

        panel_content = "\n".join(content_lines)
        panel_title = f"Evaluation History: {model_name}"
        console.print(Panel(panel_content, title=panel_title, border_style="blue"))

    else:
        # Multiple skills and models - matrix view with Panel
        _print_matrix_view(results_list, metric)

    console.print("\n[dim]--csv results.csv to export[/dim]")


@main.command("plot", hidden=True)
@click.option("-d", "--dir", "runs_dir", type=click.Path(exists=True), help="Runs directory")
@click.option("-s", "--skill", "skills", multiple=True, help="Filter by skill name(s)")
@click.option(
    "-m",
    "--model",
    "models",
    multiple=True,
    help="Filter historical run data by model(s)",
)
@click.option(
    "--metric",
    type=click.Choice(["success", "tokens"]),
    default="success",
    help="Metric to plot",
)
@click.pass_context
def plot_cmd(
    ctx: click.Context,
    runs_dir: str | None,
    skills: tuple[str, ...],
    models: tuple[str, ...],
    metric: str,
):
    """[Deprecated] Use 'upskill runs' instead."""
    console.print("[yellow]Note: 'plot' is deprecated. Use 'upskill runs' instead.[/yellow]\n")
    ctx.invoke(
        runs_cmd,
        runs_dir=runs_dir,
        skills=skills,
        models=models,
        csv_output=None,
        metric=metric,
    )


def _format_comparison_bars(
    result: EvalPlotResult,
    metric: str,
    label_field: EvalPlotLabelField = "model",
) -> str:
    """Format baseline vs with-skill comparison bars for a single result as string."""
    label = result["skill_name"] if label_field == "skill_name" else result["model"]
    has_baseline = result["has_baseline"]
    lines = [f"[bold]{label}[/bold]"]

    if metric == "success":
        with_skill_val = result["with_skill_rate"]
        with_skill_bar = _render_bar(with_skill_val)

        if has_baseline:
            baseline_val = result["baseline_rate"]
            lift = with_skill_val - baseline_val
            baseline_bar = _render_bar(baseline_val)

            lines.append(f"  baseline   {baseline_bar}  {baseline_val:>5.0%}")

            lift_str = f"+{lift:.0%}" if lift >= 0 else f"{lift:.0%}"
            lift_style = "green" if lift > 0 else "red" if lift < 0 else "dim"
            lines.append(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>5.0%}  "
                f"[{lift_style}]({lift_str})[/{lift_style}]"
            )
        else:
            lines.append(
                f"  with skill {with_skill_bar}  {with_skill_val:>5.0%}  [dim](no baseline)[/dim]"
            )
    else:  # tokens
        with_skill_val = result["with_skill_tokens"]

        if has_baseline:
            baseline_val = result["baseline_tokens"]
            max_val = max(baseline_val, with_skill_val, 1)

            baseline_bar = _render_bar(baseline_val / max_val)
            with_skill_bar = _render_bar(with_skill_val / max_val)

            savings = (baseline_val - with_skill_val) / baseline_val if baseline_val else 0
            savings_str = f"-{savings:.0%}" if savings >= 0 else f"+{-savings:.0%}"
            savings_style = "green" if savings > 0 else "red" if savings < 0 else "dim"

            lines.append(f"  baseline   {baseline_bar}  {baseline_val:>6}")
            lines.append(
                "  with skill "
                f"{with_skill_bar}  {with_skill_val:>6}  "
                f"[{savings_style}]({savings_str})[/{savings_style}]"
            )
        else:
            with_skill_bar = _render_bar(1.0 if with_skill_val > 0 else 0)
            lines.append(
                f"  with skill {with_skill_bar}  {with_skill_val:>6}  [dim](no baseline)[/dim]"
            )

    return "\n".join(lines)


def _print_comparison_bars(
    result: EvalPlotResult,
    metric: str,
    label_field: EvalPlotLabelField = "model",
) -> None:
    """Print baseline vs with-skill comparison bars for a single result."""
    console.print(_format_comparison_bars(result, metric, label_field))
    console.print()


def _print_matrix_view(results: list[EvalPlotResult], metric: str) -> None:
    """Print a matrix view for multiple skills and models."""
    # Get unique skills and models
    skills = sorted(set(r["skill_name"] for r in results))
    models = sorted(set(r["model"] for r in results))

    # Build lookup
    lookup = {(r["model"], r["skill_name"]): r for r in results}

    # Create table
    table = Table(show_header=True, title="Skill Performance Matrix")
    table.add_column("skill", style="bold")

    for model in models:
        table.add_column(model, justify="center")

    for skill in skills:
        row = [skill]
        for model in models:
            r = lookup.get((model, skill))
            if r:
                has_baseline = r["has_baseline"]
                if metric == "success":
                    with_skill = r["with_skill_rate"]
                    if has_baseline:
                        baseline = r["baseline_rate"]
                        lift = with_skill - baseline
                        lift_style = "green" if lift > 0 else "red" if lift < 0 else ""
                        cell = f"{baseline:.0%}→{with_skill:.0%}"
                        if lift_style:
                            cell = f"[{lift_style}]{cell}[/{lift_style}]"
                    else:
                        cell = f"[dim]-[/dim]→{with_skill:.0%}"
                else:
                    with_skill = r["with_skill_tokens"]
                    if has_baseline:
                        baseline = r["baseline_tokens"]
                        savings = (baseline - with_skill) / baseline if baseline else 0
                        savings_style = "green" if savings > 0 else "red" if savings < 0 else ""
                        cell = f"{baseline}→{with_skill}"
                        if savings_style:
                            cell = f"[{savings_style}]{cell}[/{savings_style}]"
                    else:
                        cell = f"[dim]-[/dim]→{with_skill}"
                row.append(cell)
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
