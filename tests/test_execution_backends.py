from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import save_json
from mcp.types import TextContent

from upskill.evaluate import evaluate_skill, load_eval_results_from_artifact_root
from upskill.executors.contracts import ExecutionHandle, ExecutionRequest, ExecutionResult
from upskill.executors.local_fast_agent import LocalFastAgentExecutor
from upskill.fast_agent_cli import build_fast_agent_command
from upskill.models import ConversationStats, ExpectedSpec, Skill, TestCase, TestResult
from upskill.result_parsing import parse_fast_agent_results


def _write_result_history(path: Path, *, assistant_text: str) -> None:
    messages = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="Do the task")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=assistant_text)],
        ),
    ]
    save_json(messages, str(path))


def _build_request(tmp_path: Path) -> ExecutionRequest:
    cards_dir = tmp_path / "cards-source"
    cards_dir.mkdir()
    (cards_dir / "evaluator.md").write_text("---\ndescription: evaluator\n---\n{{agentSkills}}\n")
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("default_model: sonnet\n")
    return ExecutionRequest(
        prompt="Do the task",
        model="haiku",
        agent="evaluator",
        fastagent_config_path=config_path,
        artifact_dir=tmp_path / "artifacts" / "run_1",
        cards_source_dir=cards_dir,
        label="test run",
        skill=Skill(
            name="write-good-prs",
            description="Write good pull request descriptions.",
            body="Use a clear structure.",
        ),
        workspace_files={"context.txt": "hello"},
        enable_shell=True,
    )


def test_build_fast_agent_command_uses_explicit_contract(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    command = build_fast_agent_command(
        request,
        skills_dir=tmp_path / "bundle" / "skills",
        results_path=tmp_path / "bundle" / "results.json",
        fast_agent_bin="fast-agent",
    )

    assert command[:2] == ["fast-agent", "go"]
    assert "--model" in command
    assert "--skills-dir" in command
    assert "--results" in command
    assert "--message" in command
    assert "--shell" in command
    assert "--quiet" in command


def test_parse_fast_agent_results_extracts_output_text(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    _write_result_history(results_path, assistant_text="Structured answer")

    parsed = parse_fast_agent_results(results_path)

    assert parsed.output_text == "Structured answer"
    assert parsed.stats.turns == 1


@pytest.mark.asyncio
async def test_local_fast_agent_executor_preserves_artifacts_and_parses_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request(tmp_path)
    executor = LocalFastAgentExecutor(fast_agent_bin="fast-agent")

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"assistant output\n", b"")

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        del kwargs
        results_index = args.index("--results") + 1
        results_path = Path(args[results_index])
        _write_result_history(results_path, assistant_text="Final answer")
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None
    assert result.output_text == "Final answer"
    assert result.raw_results_path == request.artifact_dir / "results.json"
    assert (request.artifact_dir / "request.json").exists()
    assert (request.artifact_dir / "stdout.txt").exists()
    assert (request.artifact_dir / "stderr.txt").exists()
    assert (request.artifact_dir / "workspace" / "context.txt").read_text() == "hello"
    assert (request.artifact_dir / "workspace" / "fastagent.config.yaml").exists()
    assert (request.artifact_dir / "skills" / "write-good-prs" / "SKILL.md").exists()


@pytest.mark.asyncio
async def test_local_fast_agent_executor_fails_when_results_artifact_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _build_request(tmp_path)
    executor = LocalFastAgentExecutor(fast_agent_bin="fast-agent")

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"", b"")

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        del args, kwargs
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error == "fast-agent run did not produce a results artifact."
    assert result.raw_results_path is None


@pytest.mark.asyncio
async def test_local_fast_agent_executor_normalizes_paths_and_preserves_file_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    cards_dir = Path("cards-source")
    cards_dir.mkdir()
    (cards_dir / "evaluator.md").write_text("---\ndescription: evaluator\n---\n{{agentSkills}}\n")
    config_path = Path("fastagent.config.yaml")
    config_path.write_text("default_model: sonnet\n")
    request = ExecutionRequest(
        prompt="Base prompt\n\n```context.txt\nhello\n```",
        model="haiku",
        agent="evaluator",
        fastagent_config_path=config_path,
        artifact_dir=Path("artifacts") / "run_1",
        cards_source_dir=cards_dir,
        label="test run",
        workspace_files={"context.txt": "hello"},
    )
    executor = LocalFastAgentExecutor(fast_agent_bin="fast-agent")

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"assistant output\n", b"")

    async def fake_create_subprocess_exec(*args: str, **kwargs: object) -> FakeProcess:
        cwd = kwargs["cwd"]
        assert isinstance(cwd, Path)
        assert cwd.is_absolute()
        results_index = args.index("--results") + 1
        prompt_index = args.index("--message") + 1
        skills_index = args.index("--skills-dir") + 1
        for index in (results_index, prompt_index, skills_index):
            if index == prompt_index:
                continue
            assert Path(args[index]).is_absolute()
        prompt_text = args[prompt_index]
        assert "```context.txt\nhello\n```" in prompt_text
        _write_result_history(Path(args[results_index]), assistant_text="Final answer")
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    handle = await executor.execute(request)
    result = await executor.collect(handle)

    assert result.error is None


def test_load_eval_results_from_artifact_root_reconstructs_metrics(tmp_path: Path) -> None:
    artifact_root = tmp_path / "eval"
    with_skill_dir = artifact_root / "with-skill" / "test_1"
    baseline_dir = artifact_root / "baseline" / "test_1"
    with_skill_dir.mkdir(parents=True)
    baseline_dir.mkdir(parents=True)

    test_case = TestCase(input="prompt", expected=ExpectedSpec(contains=["answer"]))
    with_skill_result = TestResult(test_case=test_case, success=True, output="answer")
    baseline_result = TestResult(test_case=test_case, success=False, output="miss")

    (with_skill_dir / "test_result.json").write_text(
        with_skill_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (baseline_dir / "test_result.json").write_text(
        baseline_result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    reconstructed = load_eval_results_from_artifact_root(
        skill_name="write-good-prs",
        model="qwen35",
        artifact_root=artifact_root,
    )

    assert reconstructed is not None
    assert reconstructed.with_skill_success_rate == 1.0
    assert reconstructed.baseline_success_rate == 0.0


@pytest.mark.asyncio
async def test_evaluate_skill_emits_per_test_progress_messages(tmp_path: Path) -> None:
    skill = Skill(
        name="write-good-prs",
        description="Write good pull request descriptions.",
        body="Use a clear structure.",
    )
    test_case = TestCase(input="prompt", expected=ExpectedSpec(contains=["answer"]))
    messages: list[str] = []

    class FakeExecutor:
        async def execute(self, request: ExecutionRequest) -> ExecutionHandle:
            request.artifact_dir.mkdir(parents=True, exist_ok=True)
            workspace_dir = request.artifact_dir / "workspace"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            task = asyncio.create_task(
                asyncio.sleep(
                    0,
                    result=ExecutionResult(
                        output_text="answer",
                        raw_results_path=None,
                        stdout_path=request.artifact_dir / "stdout.txt",
                        stderr_path=request.artifact_dir / "stderr.txt",
                        artifact_dir=request.artifact_dir,
                        workspace_dir=workspace_dir,
                        stats=ConversationStats(),
                    ),
                )
            )
            return ExecutionHandle(request=request, task=task)

        async def collect(self, handle: ExecutionHandle) -> ExecutionResult:
            return await handle.task

        async def cancel(self, handle: ExecutionHandle) -> None:
            handle.task.cancel()

    results = await evaluate_skill(
        skill,
        [test_case],
        FakeExecutor(),
        model="haiku",
        fastagent_config_path=tmp_path / "fastagent.config.yaml",
        cards_source_dir=tmp_path,
        artifact_root=tmp_path / "eval",
        progress_callback=messages.append,
    )

    assert results.with_skill_success_rate == 1.0
    assert "starting with-skill test 1/1" in messages
    assert "finished with-skill test 1/1 (ok)" in messages
