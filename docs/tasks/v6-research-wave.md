---
id: task-v6-research-wave
type: task-spec
status: done
title: W6 — generalizing improvement over v5 (research team)
parent: goal
references:
  - decisions
  - external-judges
  - possession-pipeline
---

# W6 — generalizing improvement over v5

## Request

User (2026-11-03): act as senior ML engineer, take the time for a meaningful (not quick) improvement
that generalizes to the private full match; research freely (upstream issues/PRs, papers), run
experiments, use a team of subagents, RunPod allowed.

## Product

Done means: a new production version replaces v5 only if it passes the promotion bar below, is
reproducible byte-for-byte and validated; otherwise v5 stays and the wave's negative results are
recorded. User decisions (2026-11-03): RunPod cap $15 for the wave; video ball recovery only as a
local feasibility test (match video never leaves this Mac); new dependencies allowed in scratch
experiments, production pins change only after user approval; promotion bar = D011/D014 rule.

## Promotion bar (D011/D014, unchanged)

spellF1 gain on both external judges (Metrica primary, IDSSE second), on every match clean and on
the mean of noise seeds 0/1; macroF1 not worse on average. Learned components: whole-match
exclusion for every fitted part (`harness.training_rows`). The public window is looked at only
after external selection, as a correctness/non-regression check with its interval; never tuned on
(D012). A coordinator-only held-out check on SkillCorner weak labels (never seen by workers) is
reported for any finalist as corroboration, not as a gate.

## Design

Shared harness `.orchestration/v6/harness.py` (coordinator-owned): loads the 10 external matches
clean + noisy, scores any `predict_fn(row, rows)` with the official scorer, prints per-judge means,
per-match deltas and the gate. Workers own disjoint scratch directories and never edit production.

| Stream | Hypothesis | Why it can generalize |
| --- | --- | --- |
| L learned residual | A gradient-boosted residual on top of the v5 log-odds, trained across sources with noise augmentation, recovers ownership information the rules leave (oracle-team ceiling 0.81 vs 0.60 on Metrica). | Rule evidence stays the prior; residual is bounded, team-antisymmetric, LOMO across two providers. |
| M spellF1-aware decoding | MAP Viterbi is not the minimum-risk answer for an IoU-matched spell F1; posterior (forward-backward) decoding with MBR selection or duration models places spells better. | No new features; purely a decision rule on calibrated posteriors. |
| S stoppages and restarts | (a) Stoppage onsets are mostly missed (H/A→D transitions); (b) football rules make the post-stoppage team predictable (≈70 % switch on both judges, more after the ball exits). | Rule-derived structure, measured on both judges. |
| V video ball (local) | Off-the-shelf detection recovers part of the 41 % ball gaps. | Feasibility only; aggregate recall vs ATD. |

Deferred: new external providers beyond the reserved SkillCorner check; any production wiring
before the gate.

## Outcome

v6 adopted (D015); results and every rejected recipe in `docs/PLAN.md` W6.

## Unconfirmed assumptions

- External judges' label semantics approximate the target's corrected-tracking owner (issue #17
  suggests a won header counts as a short possession, consistent with last-touch labels).
- `inject_noise` approximates the target tracker; the domain fingerprint (coordinator) checks it.
