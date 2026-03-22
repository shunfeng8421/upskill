from __future__ import annotations

from pathlib import Path

import pytest

from upskill.cli import _eval_async
from upskill.config import Config
from upskill.evaluate import apply_eval_metrics
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
    submit_calls: list[tuple[str, bool]] = []

    monkeypatch.setattr("upskill.cli.Config.load", lambda: config)

    async def fake_submit_remote_eval_jobs(**kwargs: object) -> list[str]:
        submit_calls.append((str(kwargs["model"]), bool(kwargs["run_baseline"])))
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

    assert submit_calls == [("haiku", True)]
