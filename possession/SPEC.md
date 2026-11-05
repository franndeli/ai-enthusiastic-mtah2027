---
id: possession-pipeline
type: module-design
status: draft
title: Possession pipeline (production path)
parent: architecture
references:
  - data-policy
  - technical-doc
---

# Possession pipeline

## Responsibility

Turn the competition match's tactical tracking and Smart Tagging into the submission file:
`io.load_match` → `features.compute` → `evidence.scores` → `decoder.decode` per half via
`structure.decode_halves` → `io.write_submission`, wired by `run.py` from `config/possession.json`.
v6 and v7 add the learned stages in `stack`, `dead_residual`, `pathcrf`, `handover` and `verifier`.
Stage signatures are in `architecture`; the method and its rationale are in `technical-doc`.

## Boundary

- `io.py` is the only production module that reads or writes files, and is also the
  only one that knows dataset paths (through `games.py` and `MTAH_DATA_DIR`/`MTAH_WORK_DIR`).
- `features`, `evidence`, `decoder` and `structure` are pure numpy over arrays and never print data.
- Depends on upstream `games.py` (frame table, halves, paths) and nothing else in the repo. It never
  imports `external-judges` or `evidence-experiments`. A candidate from those modules enters this
  path only after it passes the promotion gate in `goal`.
- All tunable constants live in `config/possession.json`. Fitted constants (the shape weights) are
  fitted on external data only and stored there with their provenance, never fitted on the
  competition match.
- Output is deterministic: no randomness, so the same data and config give a byte-identical file.

## Limitations

The method's accuracy limits are listed in `technical-doc` §6: hand-overs in long ball gaps,
possessions under 2 s, slow-ball stoppages, and stoppage onset following the players rather than
the referee.
