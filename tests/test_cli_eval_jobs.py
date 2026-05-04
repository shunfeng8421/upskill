from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from upskill.cli import _eval_async, _jobs_execution_options, _raise_on_execution_errors
from upskill.config import Config
from upskill.evaluate import apply_eval_metrics
from upskill.hf_jobs import JobsConfig
from upskill.logging import load_batch_summary, load_run_result
from upskill.models import (
    ConversationStats,
    EvalResults,
    ExpectedSpec,
    Skill,
    SkillRecord,
    SkillState,
    TestCase,
    TestResult,
)


def _make_eval_results(
    *,
    skill: Skill,
    model: str,
    test_cases: list[TestCase],
    run_baseline: bool,
) -> EvalResults:
    with_skill_results = [
        TestResult(
            test_case=test_case,
            success=True,
            stats=ConversationStats(total_tokens=10, turns=1),
        )
        for test_case in test_cases
    ]
    results = EvalResults(
        skill_name=skill.name,
        model=model,
        with_skill_results=with_skill_results,
    )
    if run_baseline:
        results.baseline_results = [
            TestResult(
                test_case=test_case,
                success=False,
                stats=ConversationStats(total_tokens=20, turns=1),
            )
            for test_case in test_cases
        ]
    return apply_eval_metrics(results, test_cases)


def _write_skill_fixture(skill_dir: Path) -> SkillRecord:
    record = SkillRecord(
        skill=Skill(
            name="pull-request-descriptions",
            description="Write good pull request descriptions.",
            body="Use a clear structure.",
        ),
        state=SkillState(
            tests=[
                TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"])),
                TestCase(input="prompt two", expected=ExpectedSpec(contains=["answer"])),
            ]
        ),
    )
    record.save(skill_dir)
    return record


def test_jobs_execution_options_waits_by_default() -> None:
    @click.command()
    @_jobs_execution_options(
        executor_help="Execution backend for tests",
        runs_dir_help="Runs directory for tests",
    )
    def command(
        executor: str | None,
        artifact_repo: str | None,
        wait: bool,
        jobs_timeout: str,
        jobs_flavor: str,
        jobs_secrets: str | None,
        jobs_namespace: str | None,
        max_parallel: int | None,
        runs_dir: str | None,
        log_runs: bool,
    ) -> None:
        del (
            executor,
            artifact_repo,
            jobs_timeout,
            jobs_flavor,
            jobs_secrets,
            jobs_namespace,
            max_parallel,
            runs_dir,
            log_runs,
        )
        click.echo(f"wait={wait}")

    runner = CliRunner()

    default_result = runner.invoke(command)
    assert default_result.exit_code == 0
    assert "wait=True" in default_result.output

    no_wait_result = runner.invoke(command, ["--no-wait"])
    assert no_wait_result.exit_code == 0
    assert "wait=False" in no_wait_result.output


def test_raise_on_execution_errors_surfaces_backend_failures() -> None:
    test_case = TestCase(input="prompt one", expected=ExpectedSpec(contains=["answer"]))
    results = EvalResults(
        skill_name="pull-request-descriptions",
        model="haiku",
        with_skill_results=[
            TestResult(
                test_case=test_case,
                success=False,
                error="fast-agent exited with code 1.",
            )
        ],
    )

    with pytest.raises(click.ClickException, match="execution errors") as exc_info:
        _raise_on_execution_errors(results, context="Evaluation on haiku")

    assert "with-skill test 1" in str(exc_info.value)
    assert "fast-agent exited with code 1." in str(exc_info.value)


@pytest.mark.asyncio
async def test_eval_jobs_wait_persists_simple_run_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_record = _write_skill_fixture(tmp_path / "skill")
    skill = skill_record.skill
    config = Config(runs_dir=tmp_path / "runs", fastagent_config=tmp_path / "fastagent.config.yaml")
    fake_executor = object()
    max_parallel_calls: list[int] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)
    monkeypatch.setattr("upskill.cli._build_executor", lambda *args, **kwargs: fake_executor)

    async def fake_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args
        assert kwargs["executor"] is fake_executor
        assert kwargs["operation"] == "eval"
        max_parallel = kwargs["max_parallel"]
        assert isinstance(max_parallel, int)
        max_parallel_calls.append(max_parallel)
        results = _make_eval_results(
            skill=skill,
            model=str(kwargs["model"]),
            test_cases=skill_record.state.tests,
            run_baseline=True,
        )
        return results

    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)

    await _eval_async(
        skill_path=str(tmp_path / "skill"),
        tests=None,
        models=["haiku"],
        test_gen_model=None,
        num_runs=1,
        no_baseline=False,
        verbose=False,
        executor_name="jobs",
        artifact_repo="ns/repo",
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=3,
        log_runs=True,
        runs_dir=str(config.runs_dir),
    )

    batch_folder = next(config.runs_dir.iterdir())
    summary = load_batch_summary(batch_folder)
    assert summary is not None
    assert summary.total_runs == 2
    assert summary.passed_runs == 1

    baseline_result = load_run_result(batch_folder / "run_1")
    with_skill_result = load_run_result(batch_folder / "run_2")
    assert baseline_result is not None
    assert with_skill_result is not None
    assert baseline_result.run_type == "baseline"
    assert with_skill_result.run_type == "with_skill"
    assert max_parallel_calls == [3]


@pytest.mark.asyncio
async def test_eval_uses_config_execution_defaults_when_cli_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_record = _write_skill_fixture(tmp_path / "skill")
    config = Config(
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
        executor="jobs",
        artifact_repo="ns/repo",
        num_runs=2,
        max_parallel=4,
        jobs_secrets="HF_TOKEN,ANTHROPIC_API_KEY",
        jobs_image="ghcr.io/example/custom:latest",
    )
    fake_executor = object()
    build_calls: list[str] = []
    calls: list[tuple[int, bool]] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)

    def fake_build_executor(name: str, **kwargs: object) -> object:
        build_calls.append(name)
        jobs_config = kwargs["jobs_config"]
        assert isinstance(jobs_config, JobsConfig)
        assert jobs_config.artifact_repo == "ns/repo"
        assert jobs_config.jobs_secrets == "HF_TOKEN,ANTHROPIC_API_KEY"
        assert jobs_config.jobs_image == "ghcr.io/example/custom:latest"
        return fake_executor

    async def fake_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args
        max_parallel = kwargs["max_parallel"]
        run_baseline = kwargs["run_baseline"]
        assert kwargs["executor"] is fake_executor
        assert isinstance(max_parallel, int)
        assert isinstance(run_baseline, bool)
        calls.append((max_parallel, run_baseline))
        return _make_eval_results(
            skill=skill_record.skill,
            model=str(kwargs["model"]),
            test_cases=skill_record.state.tests,
            run_baseline=False,
        )

    monkeypatch.setattr("upskill.cli._build_executor", fake_build_executor)
    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)

    await _eval_async(
        skill_path=str(tmp_path / "skill"),
        tests=None,
        models=["haiku"],
        test_gen_model=None,
        num_runs=None,
        no_baseline=False,
        verbose=False,
        executor_name=None,
        artifact_repo=None,
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets=None,
        jobs_namespace=None,
        max_parallel=None,
        log_runs=True,
        runs_dir=str(config.runs_dir),
    )

    assert build_calls == ["jobs"]
    assert calls == [(4, False), (4, False)]


@pytest.mark.asyncio
async def test_eval_cli_execution_options_override_config_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_record = _write_skill_fixture(tmp_path / "skill")
    config = Config(
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
        executor="jobs",
        num_runs=2,
        max_parallel=4,
        jobs_secrets="HF_TOKEN,ANTHROPIC_API_KEY",
    )
    fake_executor = object()
    build_calls: list[str] = []
    calls: list[tuple[int, bool]] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)

    def fake_build_executor(name: str, **kwargs: object) -> object:
        del kwargs
        build_calls.append(name)
        return fake_executor

    async def fake_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args
        max_parallel = kwargs["max_parallel"]
        run_baseline = kwargs["run_baseline"]
        assert kwargs["executor"] is fake_executor
        assert isinstance(max_parallel, int)
        assert isinstance(run_baseline, bool)
        calls.append((max_parallel, run_baseline))
        return _make_eval_results(
            skill=skill_record.skill,
            model=str(kwargs["model"]),
            test_cases=skill_record.state.tests,
            run_baseline=True,
        )

    monkeypatch.setattr("upskill.cli._build_executor", fake_build_executor)
    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)

    await _eval_async(
        skill_path=str(tmp_path / "skill"),
        tests=None,
        models=["haiku"],
        test_gen_model=None,
        num_runs=1,
        no_baseline=False,
        verbose=False,
        executor_name="local",
        artifact_repo=None,
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=1,
        log_runs=True,
        runs_dir=str(config.runs_dir),
    )

    assert build_calls == ["local"]
    assert calls == [(1, True)]


@pytest.mark.asyncio
async def test_eval_cli_jobs_secrets_override_config_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_record = _write_skill_fixture(tmp_path / "skill")
    config = Config(
        runs_dir=tmp_path / "runs",
        fastagent_config=tmp_path / "fastagent.config.yaml",
        executor="jobs",
        jobs_secrets="HF_TOKEN,ANTHROPIC_API_KEY",
    )
    fake_executor = object()
    build_calls: list[str] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)

    def fake_build_executor(name: str, **kwargs: object) -> object:
        build_calls.append(name)
        jobs_config = kwargs["jobs_config"]
        assert isinstance(jobs_config, JobsConfig)
        assert jobs_config.jobs_secrets == "HF_TOKEN,OPENAI_API_KEY"
        return fake_executor

    async def fake_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args, kwargs
        return _make_eval_results(
            skill=skill_record.skill,
            model="haiku",
            test_cases=skill_record.state.tests,
            run_baseline=True,
        )

    monkeypatch.setattr("upskill.cli._build_executor", fake_build_executor)
    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)

    await _eval_async(
        skill_path=str(tmp_path / "skill"),
        tests=None,
        models=["haiku"],
        test_gen_model=None,
        num_runs=1,
        no_baseline=False,
        verbose=False,
        executor_name="jobs",
        artifact_repo="ns/repo",
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN,OPENAI_API_KEY",
        jobs_namespace=None,
        max_parallel=1,
        log_runs=True,
        runs_dir=str(config.runs_dir),
    )

    assert build_calls == ["jobs"]


@pytest.mark.asyncio
async def test_eval_jobs_wait_persists_benchmark_run_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_record = _write_skill_fixture(tmp_path / "skill")
    skill = skill_record.skill
    config = Config(runs_dir=tmp_path / "runs", fastagent_config=tmp_path / "fastagent.config.yaml")
    fake_executor = object()

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)
    monkeypatch.setattr("upskill.cli._build_executor", lambda *args, **kwargs: fake_executor)

    calls: list[tuple[int, bool]] = []

    async def fake_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args
        artifact_root = kwargs["artifact_root"]
        assert isinstance(artifact_root, Path)
        assert kwargs["operation"] == "benchmark"
        max_parallel = kwargs["max_parallel"]
        run_baseline = kwargs["run_baseline"]
        assert isinstance(max_parallel, int)
        assert isinstance(run_baseline, bool)
        calls.append((max_parallel, run_baseline))
        results = _make_eval_results(
            skill=skill,
            model=str(kwargs["model"]),
            test_cases=skill_record.state.tests,
            run_baseline=False,
        )
        return results

    monkeypatch.setattr("upskill.cli.evaluate_skill", fake_evaluate_skill)

    await _eval_async(
        skill_path=str(tmp_path / "skill"),
        tests=None,
        models=["haiku"],
        test_gen_model=None,
        num_runs=2,
        no_baseline=True,
        verbose=False,
        executor_name="jobs",
        artifact_repo="ns/repo",
        wait=True,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=4,
        log_runs=True,
        runs_dir=str(config.runs_dir),
    )

    batch_folder = next(config.runs_dir.iterdir())
    summary = load_batch_summary(batch_folder)
    assert summary is not None
    assert summary.total_runs == 2
    assert calls == [(4, False), (4, False)]
    assert load_run_result(batch_folder / "run_1") is not None
    assert load_run_result(batch_folder / "run_2") is not None


@pytest.mark.asyncio
async def test_eval_jobs_no_wait_submits_remote_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_skill_fixture(tmp_path / "skill")
    config = Config(runs_dir=tmp_path / "runs", fastagent_config=tmp_path / "fastagent.config.yaml")
    submit_calls: list[tuple[str, bool, str]] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)

    async def fake_submit_remote_eval_jobs(**kwargs: object) -> list[str]:
        submit_calls.append(
            (
                str(kwargs["model"]),
                bool(kwargs["run_baseline"]),
                str(kwargs["operation"]),
            )
        )
        return ["evalstate/job-1", "evalstate/job-2"]

    def fail_build_executor(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("_build_executor should not be used for jobs --no-wait submission")

    async def fail_evaluate_skill(*args: object, **kwargs: object) -> EvalResults:
        del args, kwargs
        raise AssertionError("evaluate_skill should not be called for jobs --no-wait submission")

    monkeypatch.setattr("upskill.cli._submit_remote_eval_jobs", fake_submit_remote_eval_jobs)
    monkeypatch.setattr("upskill.cli._build_executor", fail_build_executor)
    monkeypatch.setattr("upskill.cli.evaluate_skill", fail_evaluate_skill)

    await _eval_async(
        skill_path=str(tmp_path / "skill"),
        tests=None,
        models=["haiku"],
        test_gen_model=None,
        num_runs=1,
        no_baseline=False,
        verbose=False,
        executor_name="jobs",
        artifact_repo="ns/repo",
        wait=False,
        jobs_timeout="2h",
        jobs_flavor="cpu-basic",
        jobs_secrets="HF_TOKEN",
        jobs_namespace=None,
        max_parallel=3,
        log_runs=True,
        runs_dir=str(config.runs_dir),
    )

    assert submit_calls == [("haiku", True, "eval")]


@pytest.mark.asyncio
async def test_eval_jobs_wait_fails_cleanly_when_artifact_repo_is_inaccessible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_skill_fixture(tmp_path / "skill")
    config = Config(runs_dir=tmp_path / "runs", fastagent_config=tmp_path / "fastagent.config.yaml")

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)
    monkeypatch.setattr(
        "upskill.cli.verify_artifact_repo_access",
        lambda _repo: (_ for _ in ()).throw(
            RuntimeError("404 Not Found\nRepository Not Found for url")
        ),
    )

    with pytest.raises(click.ClickException, match="Artifact repo is not accessible") as exc_info:
        await _eval_async(
            skill_path=str(tmp_path / "skill"),
            tests=None,
            models=["haiku"],
            test_gen_model=None,
            num_runs=1,
            no_baseline=False,
            verbose=False,
            executor_name="jobs",
            artifact_repo="evalstate/uskill-test",
            wait=True,
            jobs_timeout="2h",
            jobs_flavor="cpu-basic",
            jobs_secrets="HF_TOKEN",
            jobs_namespace=None,
            max_parallel=3,
            log_runs=True,
            runs_dir=str(config.runs_dir),
        )

    assert "Repo: evalstate/uskill-test" in str(exc_info.value)
    assert "name is wrong" in str(exc_info.value)
