---
id: goal
type: goal-and-requirements
status: draft
title: MTAH 2027 Phase 1 — ball possession labelling
references:
  - data-policy
  - decisions
---

# MTAH 2027 — spec

## Goal

Produce `evaluation/results/<name>.csv`: `frame,label`, 199,319 rows (one per tactical-video frame),
labels `H` (Barcelona in possession, ball alive), `A` (Athletic), `D` (dead ball), `X` (abstain).
Ranked on **spellF1** over the private full-match ground truth, tie-break **macroF1**.
Floor to beat: `baseline_smooth50`, spellF1 **0.2433** on the full match.

## Deliverables (SUBMISSION.md)

1. The possession file, validated with `evaluation/validate_submission.py`.
2. A **public** repository (this one, pushed to our own remote) that regenerates the file
   byte-for-byte from the provided data with one documented command, with pinned dependencies.
3. A short technical document: approach and why, assets used and dropped, handling of missing and
   noisy tracking, possession rules, limitations, how to reproduce.

## What we know about the ground truth (CHALLENGE.md)

- Derived mechanically from **corrected** tracking: `D` where it marks the ball dead, otherwise the
  team it names as owning the ball. No smoothing. The public 5 minutes (17 possessions, 7,500 graded
  frames) show what "dead" and "owning" mean in practice.
- A possession is a maximal run of one team with the ball alive. Contested or loose balls stay with
  the last team in possession. A touch, a deflection or an aerial duel without control is not a change.
- `X` on graded frames is a miss for macroF1 and "repeat your previous label" for spellF1.

## Inputs (DATA.md; gated, see DATA_POLICY.md)

Tactical video (reference timeline) and broadcast video (not synchronised), automatic tracking with
the ball for both (team-level, 46 slots, no player ids, ball missing on 41% of graded frames),
Smart Tagging (638 automatic instances), homography per frame.

## Method constraints

- Labels are produced by an algorithm only (RULES.md §1).
- No ground truth beyond the public 5 minutes.
- Any model or service in the pipeline must be accessible to the organisers. **Hosted LLMs never see
  the data** (DATA_POLICY.md), so any LLM inside the pipeline is local.
- Deterministic: same data in, same file out.
- Data-safety rules are shared; active agent configuration (`.pi/`, `.claude/`) stays local and
  ignored. Reusable guard code and tests remain optional tooling in the repo.

## Generalization acceptance gate (OpenAI review wave)

Keep the current v4 production model frozen unless a predeclared candidate passes independent
verification (v4 superseded v3c in D011).
Choose candidates using external matches only; the public window and its masks are a final
non-regression check, never a search objective. Split external matches by whole match, refitting
any learned component without the evaluated match; noise seeds are repeated measurements, not
independent matches. Previously used IDSSE matches are not a pristine holdout.
Require >= 0.03 absolute mean noisy spellF1 gain, a positive paired match-level 95% interval,
mean clean spellF1 loss <= 0.01, per-match clean/noisy-mean spellF1 loss <= 0.02, and mean
macroF1 loss <= 0.01. Require >= 0.03 gains on both target stress families per D006: 10-seed
transplant mean and mean of 5/10/20-second transition-centred masks, with no individual transition
loss > 0.02 and no native public-window regression. Report all attempted candidates and reject public-only or unstable gains.
A candidate must also pass synthetic tests, official submission validation, and byte-identical
regeneration. If evidence is insufficient, retain the current production model rather than promise
an improvement.

### Judges after W2 (v4)

The primary external judge is the Metrica sample data (`scripts/metrica_eval.py`, same provider,
last-touch label semantics, real ball gaps), clean and noisy; IDSSE is the second judge. A
production change needs a gain on both, on every match where possible, before the public window
is looked at; the public window and its stress masks remain correctness and robustness checks
(17 possessions, first five minutes). A public decrease must be explained (as in W2: stoppages)
and recorded with the decision. The earlier +0.03 / no-public-regression gate stays the bar for
learned components (D006–D010), where the public window is the only target-domain measurement.

### Judge after W7 (v7)

The noisy judges use `external.inject_tracker_noise` (D017). Compare the classifier's and the inputs'
label-free percentiles on the target and on the noisy judges before trusting a learned component: the
first W7 recipe passed every external check and was under-confident on the target, and only the tails of
those distributions showed it.

## Experiment protocols

Each research wave's predeclared protocol is its own task spec under this goal, with its outcome:
V2 `task-v2-factorized-learning`, V3 `task-v3-observation-models`, V4
`task-v4-automatic-phase-evidence`, BR2 `task-br2-tracker-robust-dead`, W7 `task-v7-learned-handover`
(all in `docs/tasks/`).
A new wave adds a task spec there; the gates above stay the standing bar.

## Open questions

- Team members (3–5 needed for the Innovation Challenge and Final Pitch eligibility).
- Name and owner of the public submission repo.
- Compute: user authorizes RunPod for substantial CPU/GPU work; no heavy local workloads.
  Use a bounded initial job and record actual resource cost/cleanup.
