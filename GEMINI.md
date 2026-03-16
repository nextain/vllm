# nextain/vllm — Fork Entry Point

This is the Nextain fork of [vllm-project/vllm](https://github.com/vllm-project/vllm).

**Fork purpose**: Add `SupportsAudioOutput` interface + MiniCPM-o audio output support for Naia OS.

## Mandatory Reads (every session)

1. **`AGENTS.md`** — upstream vLLM AI contribution policy (**highest priority, overrides everything**)
2. **`.agents/context/agents-rules.json`** — fork-specific rules (SoT)
3. **`.agents/context/project-index.yaml`** — context index
4. **`.agents/context/contributing-guide.yaml`** — vLLM contribution checklist
5. **`.agents/context/coding-conventions.yaml`** — vLLM coding conventions

Load additional context on demand: `.agents/context/audio-output-design.yaml`, `.agents/context/harness-checklist.yaml`

## Core Rule: Upstream Respect

> Every line of code must be defensible to a vLLM maintainer who is skeptical of AI-generated PRs.

Before writing any code:
1. Read the relevant upstream code first (no guessing)
2. Run duplicate PR check (`gh pr list --repo vllm-project/vllm --state open --search "..."`)
3. Understand the existing pattern and follow it exactly

Before any commit:
1. All 3 harness hooks must pass (`.claude/hooks/`)
2. `pre-commit run` must pass
3. Relevant tests must pass

Before upstream PR:
1. Complete `pre_contribution_checklist` in `.agents/context/contributing-guide.yaml`
2. PR description must include: duplicate check evidence, test results, AI disclosure

## Project Structure

```
vllm/                              # upstream vLLM source
.agents/
├── context/
│   ├── agents-rules.json          # Fork rules (SoT) ← mandatory
│   ├── project-index.yaml         # Context index ← mandatory
│   ├── contributing-guide.yaml    # vLLM contribution checklist
│   ├── coding-conventions.yaml    # vLLM coding conventions
│   ├── audio-output-design.yaml   # SupportsAudioOutput design
│   └── harness-checklist.yaml     # Harness TP/TN/FP/FN criteria
└── tests/harness/run-all.sh       # Harness test runner

.users/context/                    # Korean mirror of .agents/context/

.claude/
├── hooks/
│   ├── spdx-header-check.js       # SPDX header enforcement
│   ├── upstream-quality-check.js  # Ruff anti-pattern detection
│   └── commit-ai-guard.js         # AI disclosure + commit quality gate
└── settings.json                  # Hook registrations
```

## Key Code Paths

- **Interface to add**: `vllm/model_executor/models/interfaces.py` (follow `SupportsTranscription` pattern at line 1066)
- **Registry to update**: `vllm/model_executor/models/registry.py`
- **API server integration**: `vllm/entrypoints/openai/api_server.py`

## Harness

3 Claude Code hooks enforce upstream quality in the editor:

| Hook | Trigger | Check |
|------|---------|-------|
| `spdx-header-check.js` | Write\|Edit on `*.py` | SPDX header present and correct |
| `upstream-quality-check.js` | Write\|Edit on `vllm/**/*.py` | `print()`, bare `except:`, f-string in logger |
| `commit-ai-guard.js` | `git commit` | AI disclosure, Signed-off-by reminder, vague message block |

Run harness tests: `bash .agents/tests/harness/run-all.sh`

## Korean Mirror

Human-readable context in Korean: `.users/context/`
