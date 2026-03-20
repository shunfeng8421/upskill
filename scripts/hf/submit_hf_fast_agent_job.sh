#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_hf_fast_agent_job.sh \
    --artifact-repo <namespace/repo> \
    --skills-dir <path> \
    [--card-dir <path>] \
    [--agent <name>] \
    [--model <name>] \
    [--message <text> | --prompt-file <path> | --prompts-jsonl <path>] \
    [--flavor cpu-basic] \
    [--timeout 45m] \
    [--image python:3.13-slim-bookworm] \
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
IMAGE="python:3.13-slim-bookworm"
SECRETS="HF_TOKEN"
NAMESPACE=""
RUN_ID=""
AUTO_CONFIRM="0"
JSON_OUTPUT="0"

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
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$ARTIFACT_REPO" && -n "$SKILLS_DIR" ]] || { usage; exit 1; }
[[ -d "$SKILLS_DIR" ]] || { echo "Skills dir not found: $SKILLS_DIR" >&2; exit 1; }

input_modes=0
[[ -n "$MESSAGE" ]] && input_modes=$((input_modes+1))
[[ -n "$PROMPT_FILE" ]] && input_modes=$((input_modes+1))
[[ -n "$PROMPTS_JSONL" ]] && input_modes=$((input_modes+1))
[[ "$input_modes" -eq 1 ]] || {
  echo "Provide exactly one of --message, --prompt-file, or --prompts-jsonl" >&2
  exit 1
}

if [[ -n "$PROMPT_FILE" && ! -f "$PROMPT_FILE" ]]; then
  echo "Prompt file not found: $PROMPT_FILE" >&2
  exit 1
fi
if [[ -n "$PROMPTS_JSONL" && ! -f "$PROMPTS_JSONL" ]]; then
  echo "Prompts JSONL not found: $PROMPTS_JSONL" >&2
  exit 1
fi
if [[ -n "$CARD_DIR" && ! -d "$CARD_DIR" ]]; then
  echo "Card dir not found: $CARD_DIR" >&2
  exit 1
fi
if [[ -n "$AGENT" && -z "$CARD_DIR" ]]; then
  echo "--agent requires --card-dir" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')_fast-agent}"
echo "RUN_ID=$RUN_ID"

hf repo create "$ARTIFACT_REPO" --repo-type dataset --exist-ok >/dev/null

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

cat > "$bundle_dir/manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "artifact_repo": "$ARTIFACT_REPO",
  "skills_dir": "$SKILLS_DIR",
  "card_dir": "$CARD_DIR",
  "agent": "$AGENT",
  "model": "$MODEL",
  "mode": "$( [[ -n "$MESSAGE" ]] && echo message || ([[ -n "$PROMPT_FILE" ]] && echo prompt-file || echo prompts-jsonl) )",
  "created_at_utc": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}
JSON

tar -czf "$tmpdir/bundle.tar.gz" -C "$tmpdir" bundle

hf upload "$ARTIFACT_REPO" "$tmpdir/bundle.tar.gz" "inputs/$RUN_ID/bundle.tar.gz" \
  --repo-type dataset \
  --commit-message "inputs: $RUN_ID" >/dev/null

IFS=',' read -r -a secret_keys <<< "$SECRETS"
secret_flags=()
echo "Secrets to forward:"
for k in "${secret_keys[@]}"; do
  key="$(echo "$k" | xargs)"
  [[ -n "$key" ]] || continue
  if [[ -n "${!key:-}" ]]; then
    echo "  - $key (present locally)"
  else
    echo "  - $key (NOT set locally)"
  fi
  secret_flags+=(--secrets "$key")
done

if [[ "$AUTO_CONFIRM" != "1" ]]; then
  read -r -p "Proceed with HF Job submission? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 1; }
fi

job_cmd='
set -euo pipefail
WORK=/workspace
mkdir -p "$WORK/out"
cd "$WORK"

python -m pip install --upgrade pip
python -m pip install "huggingface_hub[cli]>=1.0" "fast-agent-mcp==0.6.2"

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
