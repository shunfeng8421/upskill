#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  submit_hf_eval_job.sh \
    --artifact-repo <namespace/repo> \
    --skill-dir <path> \
    [--tests <path/to/tests.json>] \
    [--models "haiku,sonnet"] \
    [--runs 1] \
    [--no-baseline] \
    [--verbose] \
    [--flavor cpu-basic] \
    [--timeout 2h] \
    [--image python:3.13-slim-bookworm] \
    [--upskill-ref main] \
    [--secrets HF_TOKEN,OPENAI_API_KEY] \
    [--namespace my-org] \
    [--yes] \
    [--json]
USAGE
}

ARTIFACT_REPO=""
SKILL_DIR=""
TESTS_PATH=""
MODELS=""
NUM_RUNS="1"
NO_BASELINE="0"
VERBOSE="0"
FLAVOR="cpu-basic"
TIMEOUT="2h"
IMAGE="python:3.13-slim-bookworm"
UPSKILL_REF="main"
SECRETS="HF_TOKEN"
NAMESPACE=""
RUN_ID=""
AUTO_CONFIRM="0"
JSON_OUTPUT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-repo) ARTIFACT_REPO="$2"; shift 2 ;;
    --skill-dir) SKILL_DIR="$2"; shift 2 ;;
    --tests) TESTS_PATH="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --runs) NUM_RUNS="$2"; shift 2 ;;
    --no-baseline) NO_BASELINE="1"; shift 1 ;;
    --verbose) VERBOSE="1"; shift 1 ;;
    --flavor) FLAVOR="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --upskill-ref) UPSKILL_REF="$2"; shift 2 ;;
    --secrets) SECRETS="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --yes) AUTO_CONFIRM="1"; shift 1 ;;
    --json) JSON_OUTPUT="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "$ARTIFACT_REPO" && -n "$SKILL_DIR" ]] || { usage; exit 1; }
[[ -d "$SKILL_DIR" ]] || { echo "Skill dir not found: $SKILL_DIR" >&2; exit 1; }
[[ -f "$SKILL_DIR/SKILL.md" ]] || { echo "Missing SKILL.md in $SKILL_DIR" >&2; exit 1; }
if [[ -n "$TESTS_PATH" && ! -f "$TESTS_PATH" ]]; then
  echo "Tests file not found: $TESTS_PATH" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +'%Y%m%dT%H%M%SZ')_$(basename "$SKILL_DIR")}"
echo "RUN_ID=$RUN_ID"

hf repo create "$ARTIFACT_REPO" --repo-type dataset --exist-ok >/dev/null

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

bundle_dir="$tmpdir/bundle"
mkdir -p "$bundle_dir"
cp -R "$SKILL_DIR" "$bundle_dir/skill"
cp scripts/hf/job_entrypoint.sh "$bundle_dir/job_entrypoint.sh"
chmod +x "$bundle_dir/job_entrypoint.sh"

if [[ -n "$TESTS_PATH" ]]; then
  cp "$TESTS_PATH" "$bundle_dir/tests.json"
fi
if [[ -f "upskill.config.yaml" ]]; then
  cp "upskill.config.yaml" "$bundle_dir/upskill.config.yaml"
fi
if [[ -f "fastagent.config.yaml" ]]; then
  cp "fastagent.config.yaml" "$bundle_dir/fastagent.config.yaml"
fi

cat > "$bundle_dir/manifest.json" <<JSON
{
  "run_id": "$RUN_ID",
  "artifact_repo": "$ARTIFACT_REPO",
  "skill_dir_name": "$(basename "$SKILL_DIR")",
  "models": "$MODELS",
  "runs": $NUM_RUNS,
  "no_baseline": $NO_BASELINE,
  "verbose": $VERBOSE,
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
python -m pip install "huggingface_hub[cli]>=1.0" \
  "upskill @ git+https://github.com/huggingface/upskill.git@${UPSKILL_REF}" \
  "fast-agent-mcp==0.6.2"

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
  --env "UPSKILL_REF=$UPSKILL_REF"
  --env "UPSKILL_MODELS=$MODELS"
  --env "UPSKILL_RUNS=$NUM_RUNS"
  --env "UPSKILL_NO_BASELINE=$NO_BASELINE"
  --env "UPSKILL_VERBOSE=$VERBOSE"
)

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
