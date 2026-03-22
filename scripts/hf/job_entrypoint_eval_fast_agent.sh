#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:?bundle_dir required}"
OUT_DIR="${2:?out_dir required}"

mkdir -p "$OUT_DIR/results" "$OUT_DIR/logs" "$OUT_DIR/status" "$OUT_DIR/workspaces"
cp -f "$BUNDLE_DIR/manifest.json" "$OUT_DIR/manifest.json" || true

COMMON=(fast-agent go --skills-dir "$BUNDLE_DIR/skills" --quiet)
if [[ -f "$BUNDLE_DIR/fastagent.config.yaml" ]]; then
  COMMON+=(--config-path "$BUNDLE_DIR/fastagent.config.yaml")
fi
if [[ -d "$BUNDLE_DIR/cards" ]]; then
  COMMON+=(--card "$BUNDLE_DIR/cards")
fi
if [[ -f "$BUNDLE_DIR/agent.txt" ]]; then
  COMMON+=(--agent "$(cat "$BUNDLE_DIR/agent.txt")")
fi

FAST_MODEL="${FAST_MODEL:?FAST_MODEL is required}"
overall_status=0

for request_dir in "$BUNDLE_DIR"/requests/*; do
  [[ -d "$request_dir" ]] || continue
  request_id="$(basename "$request_dir")"
  prompt_path="$request_dir/prompt.txt"
  workspace_src="$request_dir/workspace"
  shell_flag="$(cat "$request_dir/enable_shell.txt" 2>/dev/null || echo 0)"
  prompt_text="$(cat "$prompt_path")"

  workspace_tmp="$(mktemp -d)"
  if [[ -d "$workspace_src" ]]; then
    cp -a "$workspace_src/." "$workspace_tmp/" 2>/dev/null || true
  fi
  if [[ -f "$BUNDLE_DIR/fastagent.config.yaml" ]]; then
    cp -f "$BUNDLE_DIR/fastagent.config.yaml" "$workspace_tmp/fastagent.config.yaml"
  fi

  cmd=("${COMMON[@]}" --model "$FAST_MODEL" --message "$prompt_text" --results "$OUT_DIR/results/$request_id.json")
  if [[ "$shell_flag" == "1" ]]; then
    cmd+=(--shell)
  fi

  printf '%s\n' "${cmd[*]}" > "$OUT_DIR/logs/$request_id.command.txt"

  set +e
  (
    cd "$workspace_tmp"
    "${cmd[@]}" >"$OUT_DIR/logs/$request_id.out.txt" 2>"$OUT_DIR/logs/$request_id.err.txt"
  )
  status=$?
  set -e

  printf '%s\n' "$status" > "$OUT_DIR/status/$request_id.exit_code.txt"
  mkdir -p "$OUT_DIR/workspaces/$request_id"
  cp -a "$workspace_tmp/." "$OUT_DIR/workspaces/$request_id/" 2>/dev/null || true
  rm -rf "$workspace_tmp"

  if [[ "$status" -ne 0 ]]; then
    overall_status="$status"
  fi
  if [[ ! -f "$OUT_DIR/results/$request_id.json" ]]; then
    overall_status=1
  fi
done

exit "$overall_status"
