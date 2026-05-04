from __future__ import annotations

import yaml

from upskill.config import (
    UPSKILL_CONFIG_ENV,
    Config,
    find_upskill_config_path,
    resolve_upskill_config_path,
)


def test_find_upskill_config_path_uses_env_override_when_file_is_missing(
    tmp_path, monkeypatch
) -> None:
    override_path = tmp_path / "custom" / "upskill.yaml"
    monkeypatch.setenv(UPSKILL_CONFIG_ENV, str(override_path))
    monkeypatch.chdir(tmp_path)

    assert find_upskill_config_path() == override_path


def test_resolve_upskill_config_path_reports_missing_env_override(tmp_path, monkeypatch) -> None:
    override_path = tmp_path / "custom" / "upskill.yaml"
    monkeypatch.setenv(UPSKILL_CONFIG_ENV, str(override_path))

    resolution = resolve_upskill_config_path()

    assert resolution.path == override_path
    assert resolution.source == f"{UPSKILL_CONFIG_ENV} env var"
    assert resolution.exists is False


def test_config_save_uses_env_override_path_when_file_is_missing(tmp_path, monkeypatch) -> None:
    override_path = tmp_path / "custom" / "upskill.yaml"
    monkeypatch.setenv(UPSKILL_CONFIG_ENV, str(override_path))
    monkeypatch.chdir(tmp_path)

    config = Config(
        skill_generation_model="haiku",
        executor="jobs",
        artifact_repo="ns/repo",
        num_runs=4,
        max_parallel=7,
        jobs_secrets="HF_TOKEN,ANTHROPIC_API_KEY",
        jobs_image="ghcr.io/example/custom:latest",
    )
    config.save()

    assert override_path.exists()
    assert (tmp_path / "upskill.config.yaml").exists() is False

    with open(override_path, encoding="utf-8") as f:
        saved = yaml.safe_load(f) or {}

    assert saved["skill_generation_model"] == "haiku"
    assert saved["executor"] == "jobs"
    assert saved["artifact_repo"] == "ns/repo"
    assert saved["num_runs"] == 4
    assert saved["max_parallel"] == 7
    assert saved["jobs_secrets"] == "HF_TOKEN,ANTHROPIC_API_KEY"
    assert saved["jobs_image"] == "ghcr.io/example/custom:latest"


def test_config_load_reads_execution_settings(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "upskill.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "skill_generation_model: sonnet",
                "executor: jobs",
                "artifact_repo: ns/repo",
                "num_runs: 2",
                "max_parallel: 6",
                "jobs_secrets: HF_TOKEN,OPENAI_API_KEY",
                "jobs_image: ghcr.io/example/custom:latest",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = Config.load()

    assert config.skill_generation_model == "sonnet"
    assert config.executor == "jobs"
    assert config.artifact_repo == "ns/repo"
    assert config.num_runs == 2
    assert config.max_parallel == 6
    assert config.jobs_secrets == "HF_TOKEN,OPENAI_API_KEY"
    assert config.jobs_image == "ghcr.io/example/custom:latest"


def test_config_defaults_to_jobs_executor() -> None:
    config = Config()

    assert config.executor == "jobs"
