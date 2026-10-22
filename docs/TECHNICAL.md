# Technical document — MTAH 2027 Phase 1, ball possession

Status: draft, updated as results arrive. Every number here is produced by a command in this
repository on the provided data; nothing is hand-labelled or hand-corrected.

## 1. Approach in one paragraph

Per frame, the tracking is turned into *evidence* for three states — home in possession (`H`),
away in possession (`A`), dead ball (`D`) — and an offline Viterbi decoder picks the most
probable state sequence per half under a cost for changing state. The evidence is rule-based
(distances, control proxies, ball flight, stoppage signatures, Smart Tagging restarts) and
gap-aware: a missing ball is missing evidence, never a dead ball. Frames outside the graded halves
are `X`; the graded frames before the second-half kick-off are `D`. See `docs/PIPELINE.md`.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players + ball) | all evidence |
| Smart Tagging | soft dead-ball prior in set-piece windows; restart team prior |
| Tactical homography | not used (no video processing) |
| Broadcast video / ATD / homography | TBD (alignment experiment W7a) |
| Tactical video | not used |
| External: IDSSE / Sportec Open DFL (7 Bundesliga matches, CC-BY 4.0) | validation of design choices on ~2,000 labelled possessions; parameters never fitted to our match |

## 3. Handling missing and noisy tracking

TBD after the audit (`scripts/audit_data.py`) and the stress test (`scripts/stress_test.py`).

## 4. Possession rules implemented

TBD (from `possession/evidence.py`): control vs touch, loose/flying ball keeps the owner, dead
ball from out-of-bounds / stillness / tagged set pieces, restart ownership.

## 5. Validation

- Public 5-minute window (17 possessions) with 1000-draw bootstrap: pipeline correctness only.
- Gap-transplant and transition-centred stress tests on the public window.
- Leave-one-match-out on IDSSE, clean and with ATD-like noise.
- Structural checks on the full match: possession count, duration quantiles, shares.

Results table: `docs/PLAN.md`.

## 6. Limitations

TBD.

## 7. Reproduce

    scripts/setup-env.sh                     # conda env from environment.yml (pinned)
    scripts/py run.py --out evaluation/results/<team>.csv
    scripts/py evaluation/validate_submission.py evaluation/results/<team>.csv

Data: unpack the bundle as in `data/README.md` or point `MTAH_DATA_DIR` at it. Deterministic:
no randomness in the pipeline; running twice gives the identical file (checked with `cmp`).
