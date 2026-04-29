from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from upskill.ci import (
    load_eval_manifest,
    load_test_cases,
    plan_ci_suite,
    render_ci_report_markdown,
    resolve_changed_skill_dirs,
    write_ci_report,
    write_step_summary,
)
from upskill.models import (
    CiReport,
    ScenarioContribution,
    ScenarioReport,
    ScenarioVariantResult,
)

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "ci_action_repo"


def _run_git(repo: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _copy_fixture_repo(destination: Path) -> None:
    shutil.copytree(FIXTURE_REPO, destination, dirs_exist_ok=True)


def test_load_eval_manifest_and_test_cases_from_fixture() -> None:
    manifest = load_eval_manifest(FIXTURE_REPO / ".upskill" / "evals.yaml")
    cases = load_test_cases(FIXTURE_REPO / "evals" / "example.yaml")

    assert [scenario.id for scenario in manifest.scenarios] == ["fixture-scenario"]
    assert manifest.scenarios[0].skills == ["skills/example-skill"]
    assert manifest.scenarios[0].tests == "evals/example.yaml"
    assert len(cases) == 1
    assert cases[0].expected is not None
    assert cases[0].expected.contains == ["fixture", "response"]


def test_resolve_changed_skill_dirs_walks_to_nearest_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "example-skill"
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("example", encoding="utf-8")
    (refs_dir / "guide.md").write_text("details", encoding="utf-8")

    changed_skills = resolve_changed_skill_dirs(
        ["skills/example-skill/references/guide.md", "README.md"],
        working_dir=tmp_path,
    )

    assert changed_skills == ["skills/example-skill"]


def test_plan_ci_suite_selects_changed_scenarios_from_git_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _copy_fixture_repo(repo)

    _run_git(repo, "init")
    _run_git(repo, "config", "user.name", "Test User")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", "initial")

    skill_path = repo / "skills" / "example-skill" / "SKILL.md"
    skill_path.write_text(skill_path.read_text(encoding="utf-8") + "\nUpdated.\n", encoding="utf-8")
    _run_git(repo, "add", "skills/example-skill/SKILL.md")
    _run_git(repo, "commit", "-m", "update skill")

    report, selected = plan_ci_suite(
        repo / ".upskill" / "evals.yaml",
        scope="changed",
        base_ref="HEAD~1",
        working_dir=repo,
    )

    assert report.changed_files == ["skills/example-skill/SKILL.md"]
    assert report.changed_skills == ["skills/example-skill"]
    assert report.selected_scenarios == ["fixture-scenario"]
    assert [scenario.id for scenario in selected] == ["fixture-scenario"]


def test_render_ci_report_markdown_and_write_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = CiReport(
        manifest_path=".upskill/evals.yaml",
        scope="all",
        selected_scenarios=["fixture-scenario"],
        scenarios=[
            ScenarioReport(
                scenario_id="fixture-scenario",
                skills=["skills/example-skill"],
                tests_path="evals/example.yaml",
                passed=True,
                bundle=ScenarioVariantResult(
                    variant_id="bundle",
                    variant_type="bundle",
                    skills=["skills/example-skill"],
                    passed=True,
                    assertions_passed=2,
                    assertions_total=2,
                    judge_score=0.8,
                    total_tokens=42,
                ),
                ablations=[
                    ScenarioVariantResult(
                        variant_id="without-example-skill",
                        variant_type="ablation",
                        skills=[],
                        omitted_skill="skills/example-skill",
                        passed=False,
                        assertions_passed=0,
                        assertions_total=2,
                        total_tokens=21,
                    )
                ],
                baseline=ScenarioVariantResult(
                    variant_id="baseline",
                    variant_type="baseline",
                    skills=[],
                    passed=False,
                    assertions_passed=0,
                    assertions_total=2,
                    total_tokens=18,
                ),
                contributions=[
                    ScenarioContribution(
                        skill="skills/example-skill",
                        hard_score_delta=1.0,
                        judge_score_delta=0.3,
                        passed_without_skill=False,
                    )
                ],
            )
        ],
    )

    markdown = render_ci_report_markdown(report)
    assert "# upskill CI" in markdown
    assert "fixture-scenario" in markdown
    assert "without-example-skill" in markdown
    assert "skills/example-skill" in markdown

    report_path = tmp_path / "report.json"
    write_ci_report(report_path, report)
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["selected_scenarios"] == ["fixture-scenario"]

    summary_path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))
    write_step_summary(report)
    assert "fixture-scenario" in summary_path.read_text(encoding="utf-8")
