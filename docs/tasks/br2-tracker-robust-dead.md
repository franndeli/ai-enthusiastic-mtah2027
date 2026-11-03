---
id: task-br2-tracker-robust-dead
type: task-spec
status: draft
title: BR2 tracker-robust dead evidence
parent: goal
depends-on:
  - evidence-experiments
references:
  - decisions
---

## Protocol

Port the colleague branch's measurement-robustness idea, not its target calibration or smoothed
truth. Test exactly two fixed recipes: a player-only logistic activity head with fixed rule-ball
logit offset, and its 50% activity-logit blend with v4. Derive one-second player displacements,
absolute/half-relative activity and asymmetric temporal summaries within each half. Keep ownership,
ball/tag clues, confidence caps and decoder settings identical between comparator and candidate.
Refit the comparator's learned shape weights using training matches only, shared by both arms.

External selection uses unsmoothed IDSSE labels and existing Metrica event-derived labels, reported
separately. Fix whole-match development/confirmation splits, noise regimes, seeds and gates in
`docs/research/tracker_robust_dead.md` before outcomes. Treat seeds as repeats; require match-level
paired evidence, gains on both providers, standard-noise gain as well as tracker-shift robustness,
and a source-excluded fit check. Reserved matches were used historically: disclose reuse rather
than calling them pristine. No target-window calibration, selection or tie-breaking. At most one
externally selected finalist gets the existing learned-component public/stress acceptance gate;
reject rather than tune a failed finalist. No new dependencies; heavy work runs on bounded RunPod
compute with public external inputs only. No production wiring before acceptance.

## Outcome

Neither recipe deployed (D012 protocol, D013 outcome); code kept unwired
(`possession/robust_dead.py`). Evidence: `docs/research/tracker_robust_dead.md`.
