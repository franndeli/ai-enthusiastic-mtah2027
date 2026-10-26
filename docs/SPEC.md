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

Keep v3c frozen unless a predeclared candidate passes independent coordinator verification.
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
regeneration. If evidence is insufficient, retain v3c rather than promise an improvement.

### Substantive learning experiments (V2)

Test factorized learned evidence rather than only local rule changes: player-only dead/alive
classification (no frozen-ball shortcut), and externally trained ownership with fixed label
coarsening while keeping rule dead evidence. Every fitting step, including shape weights and
calibration, must exclude the outer evaluation match. Freeze recipes before results; at most two
recipes per family, no target-window selection. External event-derived or weak labels must not
be represented as independent exact frame-level possession truth. Production promotion still
requires the gates above; isolated experiments may fail without changing deployed predictions.

### Renewed evidence research (V3)

Investigate materially different observation models, not another target-window sweep. Predeclare
bounded recipes and external splits before measuring outcomes; preserve the existing promotion
gate. Test physical contact/ball-trajectory inference and source-robust temporal ownership as
separate experiments. Competition assets remain on the Mac; RunPod may process public external
data and code only, with aggregate-only logs. Local work is limited to code inspection, synthetic
tests and lightweight inference; substantial training/evaluation runs on RunPod, with bounded
spend and explicit resource cleanup. No new dependency without confirmation. Record rejected
recipes and any inability to establish a credible improvement; never claim the private score.

### Automatic phase evidence (V4)

Test the provided Smart Tagging ownership information as an additional soft observation (correlated
with tracking), never as training labels or manual corrections on the competition match. External evaluation must derive
analogous observations from actual public event feeds, not from evaluated possession ground truth;
freeze semantics, alignment and any observation-noise simulation before scores. Do not equate
manual external events with automatic target tags: disclose that domain gap and test degraded
observations. Use the unchanged promotion gate and a single externally selected target finalist.
If a valid external observation/timebase cannot be established, stop rather than manufacture an
easy benchmark. No new dependencies; heavy work remains on the bounded RunPod lease.

### Judges after W2 (v4)

The primary external judge is the Metrica sample data (`scripts/metrica_eval.py`, same provider,
last-touch label semantics, real ball gaps), clean and noisy; IDSSE is the second judge. A
production change needs a gain on both, on every match where possible, before the public window
is looked at; the public window and its stress masks remain correctness and robustness checks
(17 possessions, first five minutes). A public decrease must be explained (as in W2: stoppages)
and recorded with the decision. The earlier +0.03 / no-public-regression gate stays the bar for
learned components (D006–D010), where the public window is the only target-domain measurement.

## Open questions

- Team members (3–5 needed for the Innovation Challenge and Final Pitch eligibility).
- Name and owner of the public submission repo.
- Compute: user authorizes RunPod for substantial CPU/GPU work; no heavy local workloads.
  Use a bounded initial job and record actual resource cost/cleanup.
