SHELL := /usr/bin/env bash

# Common
ARTIFACT_REPO ?=
FLAVOR ?= cpu-basic
TIMEOUT ?= 45m
SECRETS ?= HF_TOKEN,OPENAI_API_KEY

# Fast-agent mode (lean)
SKILLS_DIR ?=
CARD_DIR ?=
FAST_AGENT ?=
FAST_MODEL ?= haiku
MESSAGE ?= Write a concise conventional commit message for: add password reset endpoint with tests.
PROMPT_FILE ?=
PROMPTS_JSONL ?=

.PHONY: \
	format format-write lint typecheck test check \
	hf-go-check hf-go-smoke hf-go-prompt hf-go-batch

format:
	uv run --extra dev scripts/format.py

format-write:
	uv run --extra dev scripts/format.py --write

lint:
	uv run --extra dev scripts/lint.py

typecheck:
	uv run --extra dev scripts/typecheck.py

test:
	uv run --extra dev pytest -v

check: format lint typecheck test

hf-go-check:
	@test -n "$(ARTIFACT_REPO)" || (echo "ARTIFACT_REPO is required" && exit 1)
	@test -n "$(SKILLS_DIR)" || (echo "SKILLS_DIR is required" && exit 1)
	@test -d "$(SKILLS_DIR)" || (echo "SKILLS_DIR not found: $(SKILLS_DIR)" && exit 1)
	@test -x scripts/hf/submit_hf_job.sh || (echo "scripts/hf/submit_hf_job.sh missing or not executable" && exit 1)
	@test -x scripts/hf/job_entrypoint_fast_agent.sh || (echo "scripts/hf/job_entrypoint_fast_agent.sh missing or not executable" && exit 1)
	@hf auth whoami >/dev/null || (echo "hf auth required: run 'hf auth login'" && exit 1)

hf-go-smoke: hf-go-check
	@cmd=(scripts/hf/submit_hf_job.sh \
	  --artifact-repo "$(ARTIFACT_REPO)" \
	  --skills-dir "$(SKILLS_DIR)" \
	  --model "$(FAST_MODEL)" \
	  --message "$(MESSAGE)" \
	  --flavor "$(FLAVOR)" \
	  --timeout "$(TIMEOUT)" \
	  --secrets "$(SECRETS)"); \
	if [[ -n "$(CARD_DIR)" ]]; then cmd+=(--card-dir "$(CARD_DIR)"); fi; \
	if [[ -n "$(FAST_AGENT)" ]]; then cmd+=(--agent "$(FAST_AGENT)"); fi; \
	echo "Running: $${cmd[*]}"; \
	"$${cmd[@]}"

hf-go-prompt: hf-go-check
	@test -n "$(PROMPT_FILE)" || (echo "PROMPT_FILE is required" && exit 1)
	@test -f "$(PROMPT_FILE)" || (echo "PROMPT_FILE not found: $(PROMPT_FILE)" && exit 1)
	@cmd=(scripts/hf/submit_hf_job.sh \
	  --artifact-repo "$(ARTIFACT_REPO)" \
	  --skills-dir "$(SKILLS_DIR)" \
	  --model "$(FAST_MODEL)" \
	  --prompt-file "$(PROMPT_FILE)" \
	  --flavor "$(FLAVOR)" \
	  --timeout "$(TIMEOUT)" \
	  --secrets "$(SECRETS)"); \
	if [[ -n "$(CARD_DIR)" ]]; then cmd+=(--card-dir "$(CARD_DIR)"); fi; \
	if [[ -n "$(FAST_AGENT)" ]]; then cmd+=(--agent "$(FAST_AGENT)"); fi; \
	echo "Running: $${cmd[*]}"; \
	"$${cmd[@]}"

hf-go-batch: hf-go-check
	@test -n "$(PROMPTS_JSONL)" || (echo "PROMPTS_JSONL is required" && exit 1)
	@test -f "$(PROMPTS_JSONL)" || (echo "PROMPTS_JSONL not found: $(PROMPTS_JSONL)" && exit 1)
	@cmd=(scripts/hf/submit_hf_job.sh \
	  --artifact-repo "$(ARTIFACT_REPO)" \
	  --skills-dir "$(SKILLS_DIR)" \
	  --model "$(FAST_MODEL)" \
	  --prompts-jsonl "$(PROMPTS_JSONL)" \
	  --flavor "$(FLAVOR)" \
	  --timeout "$(TIMEOUT)" \
	  --secrets "$(SECRETS)"); \
	if [[ -n "$(CARD_DIR)" ]]; then cmd+=(--card-dir "$(CARD_DIR)"); fi; \
	if [[ -n "$(FAST_AGENT)" ]]; then cmd+=(--agent "$(FAST_AGENT)"); fi; \
	echo "Running: $${cmd[*]}"; \
	"$${cmd[@]}"
