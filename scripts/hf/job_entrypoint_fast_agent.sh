#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="${1:?bundle_dir required}"
OUT_DIR="${2:?out_dir required}"

mkdir -p "$OUT_DIR/results" "$OUT_DIR/logs"
cp -f "$BUNDLE_DIR/manifest.json" "$OUT_DIR/manifest.json" || true

COMMON=(fast-agent go)
if [[ -f "$BUNDLE_DIR/fastagent.config.yaml" ]]; then
  COMMON+=(--config-path "$BUNDLE_DIR/fastagent.config.yaml")
fi
COMMON+=(--skills-dir "$BUNDLE_DIR/skills")
if [[ -d "$BUNDLE_DIR/cards" ]]; then
  COMMON+=(--card "$BUNDLE_DIR/cards")
fi
if [[ -n "${FAST_AGENT:-}" ]]; then
  COMMON+=(--agent "$FAST_AGENT")
fi

if [[ -f "$BUNDLE_DIR/prompts.jsonl" ]]; then
  export BUNDLE_DIR OUT_DIR
  python - <<'PY'
import json, os, subprocess, pathlib, sys
bundle = pathlib.Path(os.environ["BUNDLE_DIR"])
out = pathlib.Path(os.environ["OUT_DIR"])
base = ["fast-agent", "go"]
if (bundle / "fastagent.config.yaml").exists():
    base += ["--config-path", str(bundle / "fastagent.config.yaml")]
base += ["--skills-dir", str(bundle / "skills")]
if (bundle / "cards").exists():
    base += ["--card", str(bundle / "cards")]
agent = os.environ.get("FAST_AGENT", "")
if agent:
    base += ["--agent", agent]
default_model = os.environ.get("FAST_MODEL", "")

failures = 0
summary = []
for idx, line in enumerate((bundle / "prompts.jsonl").read_text(encoding="utf-8").splitlines(), start=1):
    line = line.strip()
    if not line:
        continue
    rec = json.loads(line)
    rid = rec.get("id") or f"case_{idx:03d}"
    msg = rec.get("message")
    if not msg:
        raise SystemExit(f"missing message at line {idx}")
    model = rec.get("model") or default_model
    result_path = out / "results" / f"{rid}.json"
    cmd = base + ["--message", msg, "--results", str(result_path)]
    if model:
        cmd += ["--model", model]
    stdout_path = out / "logs" / f"{rid}.out.txt"
    stderr_path = out / "logs" / f"{rid}.err.txt"
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, stdout=so, stderr=se)
    summary.append({"id": rid, "exit_code": proc.returncode, "model": model})
    if proc.returncode != 0:
        failures += 1

(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
if failures:
    sys.exit(1)
PY
else
  CMD=("${COMMON[@]}" --results "$OUT_DIR/results/default.json")
  if [[ -n "${FAST_MODEL:-}" ]]; then
    CMD+=(--model "$FAST_MODEL")
  fi

  if [[ -n "${FAST_MESSAGE:-}" ]]; then
    CMD+=(--message "$FAST_MESSAGE")
  elif [[ -f "$BUNDLE_DIR/prompt.txt" ]]; then
    CMD+=(--prompt-file "$BUNDLE_DIR/prompt.txt")
  else
    echo "No message/prompt input found" >&2
    exit 2
  fi

  echo "Running: ${CMD[*]}" | tee "$OUT_DIR/command.txt"
  ("${CMD[@]}") 2>&1 | tee "$OUT_DIR/logs/default.out.txt"
fi
