#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:?bundle_dir required}"
OUT_DIR="${2:?out_dir required}"

mkdir -p "$OUT_DIR"
cp -f "$BUNDLE_DIR/manifest.json" "$OUT_DIR/manifest.json" || true

SKILL_DIR="$BUNDLE_DIR/skill"
[[ -f "$SKILL_DIR/SKILL.md" ]] || { echo "SKILL.md missing in bundle skill dir"; exit 2; }

CMD=(
  upskill eval
  "$SKILL_DIR"
  --executor local
  --runs "${UPSKILL_RUNS:-1}"
  --runs-dir "$OUT_DIR/runs"
)

if [[ -f "$BUNDLE_DIR/tests.json" ]]; then
  CMD+=(--tests "$BUNDLE_DIR/tests.json")
fi

if [[ "${UPSKILL_NO_BASELINE:-0}" == "1" ]]; then
  CMD+=(--no-baseline)
fi

if [[ "${UPSKILL_VERBOSE:-0}" == "1" ]]; then
  CMD+=(-v)
fi

if [[ -n "${UPSKILL_MODELS:-}" ]]; then
  IFS=',' read -r -a models <<< "${UPSKILL_MODELS}"
  for m in "${models[@]}"; do
    mm="$(echo "$m" | xargs)"
    [[ -n "$mm" ]] && CMD+=(-m "$mm")
  done
fi

if [[ -f "$BUNDLE_DIR/upskill.config.yaml" ]]; then
  export UPSKILL_CONFIG="$BUNDLE_DIR/upskill.config.yaml"
fi
if [[ -f "$BUNDLE_DIR/fastagent.config.yaml" ]]; then
  cp -f "$BUNDLE_DIR/fastagent.config.yaml" "$OUT_DIR/fastagent.config.yaml"
fi

echo "Running: ${CMD[*]}" | tee "$OUT_DIR/command.txt"
("${CMD[@]}") 2>&1 | tee "$OUT_DIR/upskill_eval.log"
