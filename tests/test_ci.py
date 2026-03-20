from __future__ import annotations

import asyncio
from pathlib import Path

from click.testing import CliRunner

from upskill.ci import (
    load_eval_manifest,
    render_ci_report_markdown,
    run_ci_suite,
    select_scenarios,
    write_ci_report,
)
from upskill.cli import main
from upskill.models import JudgeCriterionScore, JudgeEvaluation


def _write_skill(path: Path, name: str, description: str, body: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {name}",
                f"description: {description}",
                "---",
                "",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_fixture_repo(root: Path) -> Path:
    skills_dir = root / "skills"
    _write_skill(
        skills_dir / "alpha-skill",
        "alpha-skill",
        "alpha helper",
        "Alpha bundle helper.",
    )
    _write_skill(
        skills_dir / "beta-skill",
        "beta-skill",
        "beta helper",
        "Beta bundle helper.",
    )

    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "assert_report.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from pathlib import Path",
                "import sys",
                "",
                "target = Path(sys.argv[1])",
                "content = target.read_text(encoding='utf-8').strip()",
                "if content != 'bundle ok':",
                "    raise SystemExit(f'unexpected content: {content}')",
            ]
        ),
        encoding="utf-8",
    )

    evals_dir = root / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    (evals_dir / "bundle.yaml").write_text(
        "\n".join(
            [
                "cases:",
                "  - input: write the report",
                "    output_file: report.txt",
                "    verifiers:",
                "      - type: file_exists",
                "        path: report.txt",
                "      - type: command",
                "        cmd: python scripts/assert_report.py report.txt",
            ]
        ),
        encoding="utf-8",
    )

    manifest_dir = root / ".upskill"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "evals.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "scenarios:",
                "  - id: bundle-scenario",
                "    skills:",
                "      - skills/alpha-skill",
                "      - skills/beta-skill",
                "    tests: evals/bundle.yaml",
                "    judge:",
                "      enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    return manifest_path


class _FakeEvaluatorClone:
    shell_runtime_enabled = True

    def __init__(self, parent: _FakeEvaluatorAgent) -> None:
        self._parent = parent
        self.instruction = ""
        self.message_history: list[object] = []
        self.workspace: Path | None = None

    def enable_shell(self, working_directory: Path) -> None:
        self.workspace = Path(working_directory)

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    async def set_model(self, model: str) -> None:
        self._parent.model = model

    async def send(self, user_content: str) -> str:
        assert self.workspace is not None
        alpha_path = self.workspace / "skills" / "alpha-skill" / "SKILL.md"
        beta_path = self.workspace / "skills" / "beta-skill" / "SKILL.md"
        assert alpha_path.exists()
        assert beta_path.exists()

        has_alpha = "Alpha bundle helper." in self.instruction
        has_beta = "Beta bundle helper." in self.instruction
        if has_alpha and has_beta:
            content = "bundle ok"
        elif has_alpha:
            content = "alpha only"
        elif has_beta:
            content = "beta only"
        else:
            content = "baseline"

        (self.workspace / "report.txt").write_text(content, encoding="utf-8")
        return f"{user_content}\n{content}"

    async def shutdown(self) -> None:
        return None


class _FakeEvaluatorAgent:
    def __init__(self) -> None:
        self.instruction = "Base evaluator instruction"
        self.model: str | None = None

    async def set_model(self, model: str) -> None:
        self.model = model

    async def spawn_detached_instance(self, name: str | None = None) -> _FakeEvaluatorClone:
        return _FakeEvaluatorClone(self)


class _FakeJudgeClone:
    def __init__(self, parent: _FakeJudgeAgent) -> None:
        self._parent = parent

    async def set_model(self, model: str) -> None:
        self._parent.model = model

    async def structured(self, prompt: str, schema: type[JudgeEvaluation]):
        score = 5 if "bundle ok" in prompt else 2
        result = schema(
            summary="strong" if score == 5 else "weak",
            criteria=[
                JudgeCriterionScore(
                    criterion=criterion,
                    score=score,
                    rationale="test rationale",
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
        return result, None

    async def shutdown(self) -> None:
        return None


class _FakeJudgeAgent:
    def __init__(self) -> None:
        self.model: str | None = None

    async def spawn_detached_instance(self, name: str | None = None) -> _FakeJudgeClone:
        return _FakeJudgeClone(self)


def test_manifest_selection_uses_changed_skills(tmp_path) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    manifest = load_eval_manifest(manifest_path)

    selected = select_scenarios(
        manifest,
        scope="changed",
        changed_skills=["skills/beta-skill"],
    )

    assert [scenario.id for scenario in selected] == ["bundle-scenario"]


def test_run_ci_suite_executes_bundle_and_ablations(tmp_path) -> None:
    manifest_path = _write_fixture_repo(tmp_path)
    report = asyncio.run(
        run_ci_suite(
            manifest_path,
            evaluator=_FakeEvaluatorAgent(),
            judge=_FakeJudgeAgent(),
            scope="all",
            eval_model="haiku",
            judge_model="judge-mini",
            working_dir=tmp_path,
            runs_dir=tmp_path / "runs",
        )
    )

    assert report.success is True
    assert report.selected_scenarios == ["bundle-scenario"]
    assert len(report.scenarios) == 1

    scenario = report.scenarios[0]
    assert scenario.bundle.passed is True
    assert scenario.bundle.judge_score is not None
    assert len(scenario.ablations) == 2
    assert all(item.passed is False for item in scenario.ablations)
    assert {item.skill for item in scenario.contributions} == {
        "skills/alpha-skill",
        "skills/beta-skill",
    }
    assert all(item.hard_score_delta > 0 for item in scenario.contributions)

    markdown = render_ci_report_markdown(report)
    assert "bundle-scenario" in markdown
    assert "without-alpha-skill" in markdown

    report_path = tmp_path / "report.json"
    write_ci_report(report_path, report)
    assert report_path.exists()


def test_ci_cli_forwards_options(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    manifest_path = tmp_path / "evals.yaml"
    manifest_path.write_text("scenarios: []\n", encoding="utf-8")

    async def _fake_ci_async(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("upskill.cli._ci_async", _fake_ci_async)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "ci",
            "--manifest",
            str(manifest_path),
            "--scope",
            "all",
            "--base-ref",
            "origin/release",
            "--eval-model",
            "haiku",
            "--judge-model",
            "judge-mini",
            "--summary-json",
            "report.json",
            "--runs-dir",
            "runs",
            "--fail-on-no-scenarios",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["manifest_path"] == str(manifest_path)
    assert captured["scope"] == "all"
    assert captured["base_ref"] == "origin/release"
    assert captured["eval_model"] == "haiku"
    assert captured["judge_model"] == "judge-mini"
    assert captured["summary_json"] == "report.json"
    assert captured["runs_dir"] == "runs"
    assert captured["fail_on_no_scenarios"] is True
