---
id: data-policy
type: architecture-design
status: draft
title: Data policy (rule zero)
parent: goal
---

# Data policy

## Why

OVERVIEW.md ("Use of Large Language Models"): participants *shall not upload, disclose or otherwise
make available to any third-party hosted or cloud-based LLM any videos, datasets, confidential
information or other restricted materials provided in connection with the Competition, or any
extracts thereof.* RULES.md adds: do not redistribute the data, do not publish derived data that
would reconstruct it. The terms accepted at enrolment are the final authority.

Claude (pi, Claude Code), Codex and any pi session on a hosted model are cloud LLMs. Anything a tool
prints goes into their context, which means it is sent to the provider.

## Classification

| Class | Examples | Cloud agent may |
| --- | --- | --- |
| **Public** | Everything committed upstream: docs, code, `evaluation/gt/*_public.csv`, `evaluation/results/baseline_*.csv` | read freely |
| **Gated** | Match assets: tactical/broadcast video, ATD tracking (`*_atd.txt/.xml`), homography, Smart Tagging | never read, list contents of, sample, or describe |
| **Derived, frame-level** | Features, per-frame predictions, our submission CSVs, evaluation reports, plots of frames or tracks | never read; treat as gated |
| **Aggregate** | spellF1/macroF1, possession counts, mean durations, class counts, shapes, runtimes, exceptions without values | read; this is how agents learn what happened |

When unsure, treat it as gated and ask.

## Where things live

| What | Where | In Git |
| --- | --- | --- |
| Match assets (as downloaded, untouched) | `~/.mtah2027/data/` (`MTAH_DATA_DIR`) | never |
| Intermediate and frame-level outputs | `~/.mtah2027/work/` (`MTAH_WORK_DIR`) | never |
| Candidate submissions for the scorer | `evaluation/results/<name>.csv` (gitignored upstream) | never |
| Code, docs, aggregate result tables | this repo | yes |

`~/.mtah2027` is `chmod 700`. Keep the repo's `data/` folder empty (README only) so repo-wide search
by an agent cannot hit data. Do not put data in iCloud/Dropbox-synced folders or in cloud-synced
notebooks.

## How a cloud agent works without seeing the data

1. Read the public docs (`DATA.md` describes every format) and the public starter loaders.
2. Write code that reads the data from `MTAH_DATA_DIR`.
3. Run it as `scripts/py <script>.py`. The script prints only aggregates; frame-level output goes
   to `MTAH_WORK_DIR` or `evaluation/results/`.
4. Score with `scripts/py evaluation/run_evaluation.py` (prints a scoreboard of aggregates).
5. Check data presence with `scripts/data-status.sh` (file names and sizes only).

Exceptions and tracebacks must not echo data values. Catch errors at the data boundary and report
the shape or field name, not the row.

## Enforcement

The policy applies to every developer and agent, regardless of tooling. Active `.pi/` and
`.claude/` configurations are local, ignored, and not included in fresh clones. The shared
`config/data-guard.json`, `scripts/agent_guard.py`, and tests remain available as optional guard
building blocks; they do not install hooks automatically. Existing local configurations are
preserved when untracked.

Where locally configured:

- **pi**: `.pi/extensions/data-guard.ts` (loaded once the project is trusted) blocks any tool call
  that matches the patterns in `config/data-guard.json`.
- **Claude Code**: `.claude/settings.json` has deny rules plus a `PreToolUse` hook on every tool
  running `scripts/agent_guard.py`.
- **Other tools or unconfigured clones**: no repository-installed hook. Follow `AGENTS.md` rule
  zero; do not assume enforcement exists merely because the shared guard code is present.
- Tests: `python3 -m unittest discover -s scripts -p 'test_*.py'`.

Known limits, so nobody over-trusts the guard: it checks what a tool call *names*, not what a
script *prints*. A script that prints rows of tracking defeats it. Background delegates and Fusion
children may not load project extensions. The guard is a seatbelt; the rule is the discipline.

## Local models: the only LLMs allowed to see data

Any step where a model needs to look at match content (frames, tracks, Smart Tagging rows, debugging
a specific sequence) runs on this Mac:

- **Ollama** (`/usr/local/bin/ollama`, API at `http://localhost:11434`): installed, with
  `llama3.1:8b-instruct-q4_K_M`. Pull a stronger coder or VLM model as needed. 18 GB RAM limits
  this to roughly 14B parameters at 4-bit.
- **llama.cpp** (`/opt/homebrew/bin/llama-server`): pi's `/llama` integration.
- To drive pi itself with a local model on data-touching tasks, register Ollama in
  `~/.pi/agent/models.json` (pi docs: models.md, "Configure a compatible endpoint") and pick it
  with `/model`. If a local guard is installed, it still blocks by default. Disable it consciously only for
  a local-model session.

Any model used inside the pipeline must be inspectable by the organisers (RULES.md) and described
in the technical document.

## Publishing

Before any push of the public submission repo: `git status`, then `git ls-files | grep -E
'data/|\.mp4|results/.*\.csv'` must list only `data/README.md` and `baseline_*.csv`. Never publish
frame-level exports, clipped video, or anything that would reconstruct the data.
