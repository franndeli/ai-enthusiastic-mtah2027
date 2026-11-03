---
id: task-v2-factorized-learning
type: task-spec
status: draft
title: V2 substantive learning experiments
parent: goal
references:
  - decisions
---

## Protocol

Test factorized learned evidence rather than only local rule changes: player-only dead/alive
classification (no frozen-ball shortcut), and externally trained ownership with fixed label
coarsening while keeping rule dead evidence. Every fitting step, including shape weights and
calibration, must exclude the outer evaluation match. Freeze recipes before results; at most two
recipes per family, no target-window selection. External event-derived or weak labels must not
be represented as independent exact frame-level possession truth. Production promotion still
requires the gates above; isolated experiments may fail without changing deployed predictions.

("The gates above" are the promotion gates in `docs/SPEC.md`, id `goal`.)

## Outcome

Not deployed (D008). Evidence: `docs/research/factorized_generalization.md`.
