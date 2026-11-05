---
id: task-v7-learned-handover
type: task-spec
status: done
title: W7 — learned hand-over costs and memory
parent: goal
references:
  - decisions
  - external-judges
  - possession-pipeline
---

# W7 — learned hand-over costs and memory

## Request

User (2026-11-05): read the whole repository and its rules, and produce the best spellF1 possible with
everything available, on a branch `feat/pablo.*`, without commits.

## Why this, measured first

v6's evidence on the five PathCRF-unseen matches, fresh noise seeds (`.orchestration/v7/diagnose.py`):
the true team gives 0.91-0.94 against 0.57-0.65 and the true dead frames only 0.63-0.70. The pipeline
places 190-224 of the 269-290 true team changes. So ownership is the loss, and within it the changes
that are never placed. The decoder charges one fixed 16 nats for every team change, real or not.

## Hypothesis

A classifier that estimates the log-odds of a hand-over at each frame, trained on external matches,
can price each team change individually and tell the evidence who has the ball after it. It can read
what no fixed cost can: the geometry around the ball, ball motion, and the part of PathCRF's output
v6 ignored (next possessor and the decoded possession path). It is not a retry of the rejected
emission learners (D006, D008): emissions stay rule-based, and the recipe has one memory term.

## Protocol

Selection on external data only; the match's labels and video play no part (D016).

- Training rows: noise seeds 0/1 and clean. Recipe selection: seeds 2/3 and clean, single-level
  whole-match exclusion. Confirmation: **seeds 4/5, never used for selection**, strict nesting
  (`scripts/eval_v7_external.py`: every learned part refit without the evaluated match at every level,
  v6 refit on the same rows with its own rules). The clean copies were seen during selection.
- Gate (D011/D014 and SPEC): gain on both judges, clean and noisy, on every match where possible;
  macroF1 not worse on average.
- Public window and stress tests: correctness and robustness checks of a frozen recipe, never a
  selector. It was looked at twice, once per frozen recipe (see the first attempt below).
- If the confirmation fails, v6 stays. Report every attempted recipe.

## First attempt: it transferred badly, and why

Frozen on the W6 noise (`clip(8 - 8 g, 1, 40)`, memory 1.5). Strict evaluation on seeds 4/5, against
v6 on that noise: Metrica +0.079/+0.060, IDSSE +0.044/+0.038, 10/10 cells. Provider transfer
(classifier trained on the other provider): +0.033 mean, 10/10. Public window: 0.6235 -> 0.5765,
macro 0.830 -> 0.834; full match 310 -> 289 possessions, against a rise on every external match.

The opposite move in the possession count (our own output, label-free) led to a label-free comparison
of the classifier's output on the target and on the judges: peaks above logit 0 were 2.4 per 10 minutes
against 9-19, so real hand-overs were priced above v6's fixed cost. The cause is in the inputs, and the
W6 fingerprint had missed it by comparing medians only:

| Over graded frames | Target | Judges (W6 noise) | Judges (W7 noise) |
|---|---|---|---|
| ball velocity change p50 / p99, m/s | 1.13 / 7.44 | 0.22-0.28 / 16-19 | 1.10-1.25 / 6.9-7.8 |
| ball second difference p99, cm | 5.9 | 32-46 | 6.1-6.5 |
| owner changes per 10 minutes | 23 | 30-50 | 17-23 |

The target's ball has no sharp kicks (a tracker's low-pass) and wobbles at fractions of a second.

## The W7 judge

`external.inject_tracker_noise`: the W6 matched noise plus a 6-frame Gaussian low-pass of the true ball
and 0.2 m of mid-frequency wobble (`external.TRACKER_NOISE`), chosen to reproduce the first two rows
above; the third was not calibrated. **These two numbers come from aggregate, label-free statistics of
the provided tracking**, the method of D015; they are the only thing taken from the provided game, and
everything fitted on the noised copies is fitted on external matches. On this judge:

1. Rule scalars re-selected (`scripts/tune_rules.py --tracker-noise --noisy-only`, two passes, objective
   0.4289 -> 0.4593). Largest moves: touch radius 1.25 -> 2.5 m, `w_control` 2 -> 5, `w_touch_now` 3 -> 1.
   The pass-by filter was suspected of dropping real touches on this ball; the sweep keeps it (off 0.444,
   2.5 m/s 0.4515 at that step).
2. Classifier and verifier retrained. Recipe grid on seeds 2/3: a plateau, +0.043..+0.053.

## Frozen recipe

`config/possession.json`, section `v7`: cost `clip(6 - 4 g, 1, 40)`, memory 1.5 nats, threshold -1,
tau 10 s; PathCRF weight kept at 0.5. A hand-over the classifier rates at logit -2 now costs 14 nats
(v6 charges 16), so an under-confident classifier no longer makes real changes dearer.

## Tried and not kept

Gain over the same stack with fixed costs, seeds 2/3, mean of the four groups, verifier included.

| On the W7 judge | Gain | Why not |
|---|---|---|
| Frozen recipe (a 6, b 4) | +0.0521 (10/10, min +0.0146) | **kept** |
| a 4, b 4 | +0.0529 (10/10, min +0.0123) | same plateau, worse minimum |
| a 8, b 8 (the first attempt's) | +0.0459 (10/10, min +0.0092) | worse |
| costs capped at 16, 20 or 24 | +0.0434 to +0.0463 | no gain over the plateau |
| no memory | +0.0130 (9/10) | the memory gives most of the gain |
| classifier trained on noisy copies only | +0.0364 | worse |
| PathCRF weight 0.75 / 1.0 / 1.5 | +0.050 / +0.043 / +0.032 | worse |

| On the W6 noise | Gain | Why not |
|---|---|---|
| + learned alive-to-dead cost | +0.002 over the base | inside the noise, one more model |
| + classifier signals in the verifier | +0.002 over the base | inside the noise |
| blend with a ten-match classifier without PathCRF | none | no gain |
| team-emission scale 0.3-0.7 | IDSSE noisy -0.044..-0.001 | breaks a judge |

## Outcome

Adopted (D017). Evidence in `docs/PLAN.md` W7.
