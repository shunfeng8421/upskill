from __future__ import annotations

from click.testing import CliRunner

from upskill.cli import main
from upskill.evaluate import build_default_judge_evaluation, rank_candidate_results
from upskill.logging import load_ranking_summary, write_ranking_summary
from upskill.models import (
    CandidateEvalResult,
    JudgeCriterionScore,
    JudgeEvaluation,
    RankedSkillBatch,
    Skill,
)


def _make_skill(name: str) -> Skill:
    return Skill(name=name, description=f"{name} desc", body=f"# {name}")


def _judge_eval(score: int, summary: str) -> JudgeEvaluation:
    return JudgeEvaluation(
        summary=summary,
        criteria=[
            JudgeCriterionScore(
                criterion=criterion,
                score=score,
                rationale=f"{criterion} rationale",
            )
            for criterion in (
                "instruction_quality",
                "helpfulness",
                "robustness",
                "concision",
                "generalizability",
            )
        ],
    )


def test_rank_candidate_results_prefers_hard_score_over_judge_score() -> None:
    strong = CandidateEvalResult(
        candidate_id="candidate-1",
        skill=_make_skill("strong-skill"),
        assertions_passed=9,
        assertions_total=10,
        hard_score=0.9,
        judge_score=0.4,
        average_tokens=200,
    )
    flashy = CandidateEvalResult(
        candidate_id="candidate-2",
        skill=_make_skill("flashy-skill"),
        assertions_passed=6,
        assertions_total=10,
        hard_score=0.6,
        judge_score=1.0,
        average_tokens=100,
    )

    ranking = rank_candidate_results(
        "task",
        [flashy, strong],
        skill_generation_model="sonnet",
        evaluation_model="sonnet",
        judge_model="haiku",
        judge_strategy="pointwise",
        tests=[],
    )

    assert ranking.winner is not None
    assert ranking.winner.candidate.candidate_id == "candidate-1"
    assert ranking.ranked_results[1].candidate.hard_gate_failed is True


def test_build_default_judge_evaluation_is_neutral() -> None:
    result = build_default_judge_evaluation("fallback")

    assert result.summary == "fallback"
    assert len(result.criteria) == 5
    assert result.total_score == 15
    assert result.normalized_score == 0.6


def test_ranking_summary_round_trip(tmp_path) -> None:
    ranking = RankedSkillBatch(
        task="task",
        skill_generation_model="sonnet",
        evaluation_model="haiku",
        judge_model="haiku",
        candidate_count=1,
    )

    write_ranking_summary(tmp_path, ranking)
    loaded = load_ranking_summary(tmp_path)

    assert loaded is not None
    assert loaded.model_dump(mode="json") == ranking.model_dump(mode="json")


def test_generate_cli_forwards_judge_options(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    async def _fake_generate_async(*args):
        (
            task,
            examples,
            from_skill,
            from_trace,
            model,
            test_gen_model,
            output,
            no_eval,
            eval_model,
            candidates,
            judge_model,
            rank_with_judge,
            judge_strategy,
            runs_dir,
            log_runs,
        ) = args
        captured.update(
            {
                "task": task,
                "examples": examples,
                "from_skill": from_skill,
                "from_trace": from_trace,
                "model": model,
                "test_gen_model": test_gen_model,
                "output": output,
                "no_eval": no_eval,
                "eval_model": eval_model,
                "candidates": candidates,
                "judge_model": judge_model,
                "rank_with_judge": rank_with_judge,
                "judge_strategy": judge_strategy,
                "runs_dir": runs_dir,
                "log_runs": log_runs,
            }
        )

    monkeypatch.setattr("upskill.cli._generate_async", _fake_generate_async)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "generate",
            "rank skills",
            "--candidates",
            "3",
            "--judge-model",
            "haiku",
            "--rank-with-judge",
            "--judge-strategy",
            "pointwise",
            "--runs-dir",
            str(tmp_path),
            "--no-log-runs",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["task"] == "rank skills"
    assert captured["candidates"] == 3
    assert captured["judge_model"] == "haiku"
    assert captured["rank_with_judge"] is True
    assert captured["judge_strategy"] == "pointwise"
    assert captured["log_runs"] is False
