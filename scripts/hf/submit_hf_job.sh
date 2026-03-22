#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_IMAGE="ghcr.io/astral-sh/uv:python3.13-bookworm"

usage() {
  cat <<'USAGE'
Usage:
  submit_hf_job.sh \
    --artifact-repo <namespace/repo> \
    --skills-dir <path> \
    [--card-dir <path>] \
    [--agent <name>] \
    [--model <name>] \
    [--message <text> | --prompt-file <path> | --prompts-jsonl <path>] \
    [--flavor cpu-basic] \
    [--timeout 45m] \
    [--image ghcr.io/astral-sh/uv:python3.13-bookworm] \
    [--secrets HF_TOKEN,OPENAI_API_KEY] \
    [--namespace my-org] \
    [--yes] \
    [--json]

Notes:
  - Artifacts are stored in dataset repo under inputs/<run_id>/ and outputs/<run_id>/
  - prompts-jsonl mode expects one JSON object per line:
      {"id":"case1","message":"...","model":"haiku"}
USAGE
}

fail() {
  echo "$*" >&2
  exit 1
}

trim() {
  xargs <<<"$1"
}

prepare_secret_flags() {
  IFS=',' read -r -a secret_keys <<< "$SECRETS"
  secret_flags=()
  echo "Secrets to forward:"
  for raw_key in "${secret_keys[@]}"; do
    key="$(trim "$raw_key")"
    [[ -n "$key" ]] || continue
    if [[ -n "${!key:-}" ]]; then
      echo "  - $key (present locally)"
    else
      echo "  - $key (NOT set locally)"
    fi
    secret_flags+=(--secrets "$key")
  done
}

check_artifact_repo() {
  hf download "$ARTIFACT_REPO" --repo-type dataset --dry-run --quiet >/dev/null || \
    fail "Artifact repo $ARTIFACT_REPO is not accessible. Create it first and ensure your current Hugging Face credentials can access it."
}

submit_bundle_job() {
  check_artifact_repo

  tar -czf "$tmpdir/bundle.tar.gz" -C "$tmpdir" bundle
  hf upload "$ARTIFACT_REPO" "$tmpdir/bundle.tar.gz" "inputs/$RUN_ID/bundle.tar.gz" \
    --repo-type dataset \
    --commit-message "inputs: $RUN_ID" >/dev/null

  prepare_secret_flags

  if [[ "$AUTO_CONFIRM" != "1" ]]; then
    read -r -p "Proceed with HF Job submission? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || fail "Cancelled."
  fi

  ns_flags=()
  if [[ -n "$NAMESPACE" ]]; then
    ns_flags+=(--namespace "$NAMESPACE")
  fi

  job_id="$(
    hf jobs run \
      --detach \
      --flavor "$FLAVOR" \
      --timeout "$TIMEOUT" \
      "${ns_flags[@]}" \
      "${secret_flags[@]}" \
      "${env_flags[@]}" \
      -- \
      "$IMAGE" \
      bash -lc "$job_cmd"
  )"

  job_id="$(echo "$job_id" | tail -n 1 | xargs)"

  if [[ "$JSON_OUTPUT" == "1" ]]; then
    cat <<JSON
{"job_id":"$job_id","run_id":"$RUN_ID","artifact_repo":"$ARTIFACT_REPO"}
JSON
  else
    echo "JOB_ID=$job_id"
    echo "RUN_ID=$RUN_ID"
    echo "ARTIFACT_REPO=$ARTIFACT_REPO"
  fi
}

ARTIFACT_REPO=""
SKILLS_DIR=""
CARD_DIR=""
AGENT=""
MODEL=""
MESSAGE=""
PROMPT_FILE=""
PROMPTS_JSONL=""
FLAVOR="cpu-basic"
TIMEOUT="45m"
IMAGE="$DEFAULT_IMAGE"
SECRETS="HF_TOKEN"
NAMESPACE=""
RUN_ID=""
AUTO_CONFIRM="0"
JSON_OUTPUT="0"
tmpdir=""
job_cmd=""
secret_flags=()
env_flags=()
ns_flags=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-repo) ARTIFACT_REPO="$2"; shift 2 ;;
    --skills-dir) SKILLS_DIR="$2"; shift 2 ;;
    --card-dir) CARD_DIR="$2"; shift 2 ;;
    --agent) AGENT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --message) MESSAGE="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --prompts-jsonl) PROMPTS_JSONL="$2"; shift 2 ;;
    --flavor) FLAVOR="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --secrets) SECRETS="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --yes) AUTO_CONFIRM="1"; shift 1 ;;
    --json) JSON_OUTPUT="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown arg: $1" ;;
  esac
done

[[ -n "$ARTIFACT_REPO" && -n "$SKILLS_DIR" ]] || {
  usage
  exit 1
}
[[ -d "$SKILLS_DIR" ]] || fail "Skills dir not found: $SKILLS_DIR"

input_modes=0
[[ -n "$MESSAGE" ]] && input_modes=$((input_modes + 1))
[[ -n "$PROMPT_FILE" ]] && input_modes=$((input_modes + 1))
[[ -n "$PROMPTS_JSONL" ]] && input_modes=$((input_modes + 1))
[[ "$input_modes" -eq 1 ]] || fail "Provide exactly one of --message, --prompt-file, or --prompts-jsonl"

if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
  fail "Prompt file not found: $PROMPT_FILE"
fi
if [[ -n "$PROMPTS_JSONL" && ! -f "$PROMPTS_JSONL" ]]; then
  fail "Prompts JSONL not found: $PROMPTS_JSONL"
fi
if [[ -n "$CARD_DIR" && ! -d "$CARD_DIR" ]]; then
  fail "Card dir not found: $CARD_DIR"
fi
if [[ -n "$AGENT" && -z "$CARD_DIR" ]]; then
  fail "--agent requires --card-dir"
fi

RUN_ID="${RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')_fast-agent}"
echo "RUN_ID=$RUN_ID"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
bundle_dir="$tmpdir/bundle"
mkdir -p "$bundle_dir"

cp -R "$SKILLS_DIR" "$bundle_dir/skills"
cp scripts/hf/job_entrypoint_fast_agent.sh "$bundle_dir/job_entrypoint.sh"
chmod +x "$bundle_dir/job_entrypoint.sh"

if [[ -n "$CARD_DIR" ]]; then
  cp -R "$CARD_DIR" "$bundle_dir/cards"
fi
if [[ -n "$PROMPT_FILE" ]]; then
  cp "$PROMPT_FILE" "$bundle_dir/prompt.txt"
fi
if [[ -n "$PROMPTS_JSONL" ]]; then
  cp "$PROMPTS_JSONL" "$bundle_dir/prompts.jsonl"
fi
if [[ -f "fastagent.config.yaml" ]]; then
  cp "fastagent.config.yaml" "$bundle_dir/fastagent.config.yaml"
fi

mode_name="prompts-jsonl"
if [[ -n "$MESSAGE" ]]; then
  mode_name="message"
elif [[ -n "$PROMPT_FILE" ]]; then
  mode_name="prompt-file"
fi

cat > "$bundle_dir/manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "artifact_repo": "$ARTIFACT_REPO",
  "skills_dir": "$SKILLS_DIR",
  "card_dir": "$CARD_DIR",
  "agent": "$AGENT",
  "model": "$MODEL",
  "mode": "$mode_name",
  "created_at_utc": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}
JSON

job_cmd='
set -euo pipefail
WORK=/workspace
mkdir -p "$WORK/out"
cd "$WORK"

uv pip install --system "huggingface_hub[cli]==1.7.2" "fast-agent-mcp==0.6.2"

hf download "$ARTIFACT_REPO" "inputs/$RUN_ID/bundle.tar.gz" --repo-type dataset --local-dir "$WORK"
tar -xzf "$WORK/inputs/$RUN_ID/bundle.tar.gz" -C "$WORK"

set +e
bash "$WORK/bundle/job_entrypoint.sh" "$WORK/bundle" "$WORK/out"
status=$?
set -e

echo "$status" > "$WORK/out/exit_code.txt"

hf upload "$ARTIFACT_REPO" "$WORK/out" "outputs/$RUN_ID" \
  --repo-type dataset \
  --commit-message "outputs: $RUN_ID (exit=$status)"

exit "$status"
'

env_flags=(
  --env "ARTIFACT_REPO=$ARTIFACT_REPO"
  --env "RUN_ID=$RUN_ID"
  --env "FAST_MODEL=$MODEL"
  --env "FAST_AGENT=$AGENT"
)
if [[ -n "$MESSAGE" ]]; then
  env_flags+=(--env "FAST_MESSAGE=$MESSAGE")
fi

submit_bundle_job
