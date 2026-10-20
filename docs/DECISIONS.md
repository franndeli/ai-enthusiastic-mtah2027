# MTAH 2027 — decision log

Record consequential decisions concisely: context, decision, consequence, and revisit condition.
Supersede entries explicitly; do not erase historical reasoning.

| ID / date | Decision | Consequence | Revisit when |
| --- | --- | --- | --- |
| D001 / 2026-10-20 | Cloud-hosted LLM agents (Claude, Codex, pi on hosted models) work on code and public material only. Gated data and frame-level derived outputs never enter their context; they learn from aggregate script output. | Data lives in `~/.mtah2027` outside the repo; a shared guard (`config/data-guard.json`, `scripts/agent_guard.py`) blocks pi and Claude Code tool calls that name data paths; any LLM that looks at data runs locally. | The organisers clarify the terms differently in writing. |
| D004 / 2026-10-20 | Active `.pi/` and `.claude/` configurations stay local and ignored; retain shared data policy and reusable guard code/tests. Supersedes D001's implication that hooks ship enabled for everyone, not its data restrictions. | Developers choose their tooling; fresh clones have no installed guard hooks. Existing local configurations remain intact. | The team explicitly agrees to maintain shared tool-specific configurations. |
| D002 / 2026-10-20 | Work in a clone of the official repository (`upstream` remote), keeping its scorer, loaders and layout; our code is added alongside. | Upstream fixes merge easily, the scorer we run is the official one, and the public submission repo is this repo pushed to our own remote. | Upstream restructures incompatibly. |
| D003 / 2026-10-20 | One project Conda env (`./.conda-env`, Python 3.11) mirroring `requirements-starter.txt`; code runs through `scripts/py`. | Isolated from base; data paths injected by the wrapper, never typed by agents. Exact pins are exported before submission for reproducibility. | A dependency needs a different Python. |
