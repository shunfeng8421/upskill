#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TASK="${TASK:-write a good pull request description}"
MODEL="${MODEL:-qwen35}"
GENERATE_MODEL="${GENERATE_MODEL:-$MODEL}"
TEST_GEN_MODEL="${TEST_GEN_MODEL:-opus}"
START_AT="${START_AT:-prepare}"
ARTIFACT_REPO="${ARTIFACT_REPO:?Set ARTIFACT_REPO to <namespace>/upskill-evals}"
JOBS_SECRETS="${JOBS_SECRETS:?Set JOBS_SECRETS, e.g. HF_TOKEN,OPENROUTER_API_KEY}"
JOBS_TIMEOUT="${JOBS_TIMEOUT:-45m}"
JOBS_FLAVOR="${JOBS_FLAVOR:-cpu-basic}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/.smoke-test}"
SKILL_OUTPUT="${SKILL_OUTPUT:-$OUT_ROOT/generated-skill}"
LOCAL_RUNS_DIR="${LOCAL_RUNS_DIR:-$OUT_ROOT/local-runs}"
REMOTE_RUNS_DIR="${REMOTE_RUNS_DIR:-$OUT_ROOT/remote-runs}"

mkdir -p "$OUT_ROOT"

if [[ "$START_AT" != "prepare" && "$START_AT" != "remote" && "$START_AT" != "local" ]]; then
  echo "START_AT must be one of: prepare, remote, local" >&2
  exit 1
fi

has_prepared_skill=0
if [[ -f "$SKILL_OUTPUT/SKILL.md" && -f "$SKILL_OUTPUT/skill_meta.json" ]]; then
  has_prepared_skill=1
fi

echo "== Model secret check =="
fast-agent check models --for-model "$MODEL" --json || true

if [[ "$START_AT" == "prepare" ]]; then
  echo
  echo "== Prepare skill + tests (no eval) =="
  rm -rf "$SKILL_OUTPUT"
  mkdir -p "$(dirname "$SKILL_OUTPUT")"
  export SMOKE_TASK="$TASK"
  export SMOKE_GENERATE_MODEL="$GENERATE_MODEL"
  export SMOKE_TEST_GEN_MODEL="$TEST_GEN_MODEL"
  export SMOKE_SKILL_OUTPUT="$SKILL_OUTPUT"
  uv run python - <<'PY'
import asyncio
import os
from pathlib import Path

from upskill.cli import _fast_agent_context, _set_agent_model
from upskill.config import Config
from upskill.generate import generate_skill, generate_tests


async def main() -> None:
    task = os.environ["SMOKE_TASK"]
    generate_model = os.environ["SMOKE_GENERATE_MODEL"]
    test_gen_model = os.environ["SMOKE_TEST_GEN_MODEL"]
    output_path = Path(os.environ["SMOKE_SKILL_OUTPUT"])
    config = Config.load()

    async with _fast_agent_context(config) as agent:
        await _set_agent_model(agent.skill_gen, generate_model)
        skill = await generate_skill(
            task=task,
            generator=agent.skill_gen,
            model=generate_model,
        )
        await _set_agent_model(agent.test_gen, test_gen_model)
        tests = await generate_tests(
            task=task,
            generator=agent.test_gen,
            model=test_gen_model,
        )
    skill.save(output_path, tests=tests)
    print(f"Prepared skill with tests at {output_path}")


asyncio.run(main())
PY
  has_prepared_skill=1
else
  echo
  echo "== Reusing prepared skill =="
  if [[ "$has_prepared_skill" != "1" ]]; then
    echo "Prepared skill not found at $SKILL_OUTPUT" >&2
    echo "Run with START_AT=prepare first." >&2
    exit 1
  fi
  echo "Using $SKILL_OUTPUT"
fi

if [[ "$START_AT" == "prepare" || "$START_AT" == "remote" ]]; then
  echo
  echo "== Remote eval via HF Jobs =="
  uv run upskill eval "$SKILL_OUTPUT" \
    --executor jobs \
    --artifact-repo "$ARTIFACT_REPO" \
    -m "$MODEL" \
    --wait \
    --jobs-timeout "$JOBS_TIMEOUT" \
    --jobs-flavor "$JOBS_FLAVOR" \
    --jobs-secrets "$JOBS_SECRETS" \
    --runs-dir "$REMOTE_RUNS_DIR"
fi

if [[ "$START_AT" == "prepare" || "$START_AT" == "remote" || "$START_AT" == "local" ]]; then
  echo
  echo "== Local eval via local shell-out executor =="
  uv run upskill eval "$SKILL_OUTPUT" \
    --executor local \
    -m "$GENERATE_MODEL" \
    --runs-dir "$LOCAL_RUNS_DIR"
fi

echo
echo "Smoke test complete."
echo "  Skill output:   $SKILL_OUTPUT"
echo "  Local runs:     $LOCAL_RUNS_DIR"
echo "  Remote runs:    $REMOTE_RUNS_DIR"
