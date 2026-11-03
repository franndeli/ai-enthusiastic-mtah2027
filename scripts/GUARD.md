---
id: data-guard
type: module-design
status: draft
title: Agent data guard (optional tooling)
parent: architecture
references:
  - data-policy
---

# Agent data guard

## Responsibility

Block an agent tool call that names gated data paths, before it runs.
`scripts/agent_guard.py` reads one tool call as JSON on stdin and applies the patterns in
`config/data-guard.json`. It exits 0 to allow the call and 2 to block it, with the reason on stderr.
Tests: `python3 -m unittest discover -s scripts -p 'test_*.py'`.

## Boundary

- Standard library only, so it runs under any `python3` without the project env. It fails closed.
- Independent of the `possession` package and of the data itself.
- Ships as reusable code only. The hooks that call it (`.pi/extensions/data-guard.ts`,
  `.claude/settings.json`) are local and ignored (D004).
- It checks what a call *names*, not what a script *prints*. Rule zero in `data-policy` still
  applies, with or without the guard.
