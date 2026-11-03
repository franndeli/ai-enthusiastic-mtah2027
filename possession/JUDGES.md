---
id: external-judges
type: module-design
status: draft
title: External judges (IDSSE, Metrica)
parent: architecture
depends-on:
  - possession-pipeline
references:
  - decisions
---

# External judges

## Responsibility

Supply labelled matches other than the competition match, so that design choices are made
without the target. `external.py` loads the IDSSE / Sportec Open DFL matches and injects noise
that mimics our tracking: ball dropout drawn from our gap statistics, jitter, and fragmented or
duplicated tracks. `metrica.py` loads the Metrica sample matches and derives last-touch labels from
their events. Both return a `possession.io.Match` plus H/A/D/X codes, so the production stages run
on them unchanged. `external.inject_matched_noise` (W6) is the target-matched noise: smooth
low-frequency ball error (`MATCHED_NOISE`: Metrica 0.75 m, IDSSE 0.3 m), ball dropout with the
target's gap statistics and player dropout; the older `inject_noise` (iid 0.3 m jitter) is kept
for reproducing pre-W6 numbers.
Fetching, preparing and scoring are done by `scripts/fetch_*.py`, `idsse_prepare.py`,
`idsse_eval.py`, `metrica_eval.py` and `experiment.py`, with the official scorer in `evaluation/`.
These rule evaluators default to the rule-only `config/possession_v5.json`; the v6 learned stack
is trained on these same matches, so its external scores need whole-match exclusion (W6 harness,
`docs/PLAN.md` W6).

Metrica is the primary judge and IDSSE the second (D011). How each judge's results are used for
promotion is set by the gate in `goal`.

## Boundary

- Reads only public external data, downloaded into `MTAH_WORK_DIR` and cached there. Never reads
  competition assets, and never produces labels for the competition match.
- Every fitting step evaluated on a match must exclude that whole match. Noise seeds are repeated
  measurements of one match, not independent matches.

## Limitations

- These labels do not match the target semantics. IDSSE is finer-grained, with more and shorter
  spells and a higher D share than our public window, so it is used for relative decisions only.
  Metrica labels are derived from events, not from corrected tracking. Neither is the target's
  corrected-tracking truth.
- The available matches have been reused across waves. None is a pristine holdout, and results
  must disclose that reuse.
