# W7 — third-provider check of v6 (SkillCorner, 20 matches), 2026-11-09/10

**Outcome: v6 stays, now confirmed on a provider nothing was built on.** One retraining
candidate was tested and rejected. Protocol and bar: `docs/tasks/v7-third-provider.md`.
Aggregates only; external public data only. Scratch: `.orchestration/v7/`.

## 1. The judge

`scripts/fetch_skillcorner.py` (pinned commit, 80 files hash-checked against
`config/skillcorner_manifest.json`, 1.9 GB) and `possession/skillcorner.py` (12 synthetic tests).
Broadcast tracking at 10 fps, resampled to 25 fps; the ball counts as seen only where the provider
marks it detected; off-camera players are the provider's extrapolation. No noise is injected.
`scripts/py scripts/skillcorner_audit.py`:

| | SkillCorner (20 matches) | competition match |
|---|---|---|
| ball seen, graded frames | 59.4 % | 59.3 % |
| frames in ball gaps 3-10 s / 10-30 s / ≥ 30 s | 5.8 / 14.7 / 18.7 % | 6.0 / 16.9 / 17.8 % |
| players seen per frame | 16.3 | 20.5 |
| ball seen on dead / alive frames | 0.30 / 0.90 | public window 0.66 / 0.94 |
| possessions per match, median, share < 2 s | 247, 7 s, 16 % | public window: 17 in 5 min, 2 under 2 s |

Labels are ours, from provider events, in two variants. *event* (primary; construction first
built on the teammate branch): team of the last player possession carried forward, D from game
interruptions. We fixed one defect of that construction: the restart in flight was given to the
previous owner, which created a spurious short possession after a stoppage whenever the restart
went to the other team; it now belongs to the team taking it. *phase* (secondary): provider
phases, D elsewhere. The two agree on 90.7 % of frames for dead/alive and 96.8 % for the team.
Both are proxies, not corrected tracking.

Split by sorted id: DEV = first ten (nine touched before as weak corroboration), TEST = last ten,
never used by main until the single look in section 4.

## 2. Where v6 loses (before SkillCorner)

`.orchestration/v7/diag.py`, nested v6 on the five PathCRF-unseen Metrica/IDSSE matches (it
reproduces `scripts/eval_v6_external.py` to four decimals):

| | Metrica clean / noisy | IDSSE clean / noisy |
|---|---|---|
| v6 | 0.654 / 0.585 | 0.660 / 0.590 |
| with the true dead frames | 0.701 / 0.633 | 0.697 / 0.670 |
| with the true team on alive frames | 0.861 / 0.841 | 0.924 / 0.872 |

Recall at IoU 0.5 by true possession length: under 2 s 0.14-0.23, 2-4 s 0.43-0.53, 4-8 s
0.63-0.80, longer 0.76-0.98. Of the missed possessions under 4 s, 53-63 % are swallowed by the
other team's spell and 29-36 % sit inside a too-long predicted spell. Frame-level team accuracy
is already 0.92-0.95 with the ball seen. The loss is short possessions.

## 3. v6 stage by stage on SkillCorner DEV (`.orchestration/v7/sc_e1.py dev`)

Production models, none of them trained or tuned on SkillCorner. spellF1, event labels (phase
labels in brackets), 90 % paired match bootstrap:

| step | gain | matches up |
|---|---|---|
| v5 config → rules re-tuned in D016 | +0.008 [−0.001, +0.018] (+0.012) | 7/10 |
| + fresh-exit clue | +0.006 [+0.005, +0.008] (+0.006) | 10/10 |
| + dead residual | +0.034 [+0.027, +0.040] (+0.036), macroF1 +0.10 | 10/10 |
| + PathCRF x0.5 | +0.041 [+0.034, +0.049] (+0.042) | 10/10 |
| + verifier | +0.004 [−0.001, +0.009] (+0.002) | 5/10 |
| **v5 config → v6** | **0.444 → 0.537, +0.093 [+0.084, +0.103]** (0.470 → 0.568) | 10/10 |

PathCRF weight (`w_sweep.py`, no verifier): 0 / 0.25 / 0.5 / 0.75 / 1.0 / 1.5 give 0.492 / 0.525 /
**0.534** / 0.528 / 0.523 / 0.513 on DEV; on Metrica/IDSSE 0.5 and 0.75 are level and both ends
lose. The weight chosen in W6 is the optimum on the third provider too.

Dead ball is where SkillCorner differs: v6 calls 32 % of frames dead against 50 % true, and the
true dead frames alone would lift it from 0.534 to 0.719 (`sc_diag.py`). That is a property of
broadcast tracking, not of the target: inside ball gaps of 10 s or more SkillCorner's frames are
91-93 % dead and v6 calls 57-65 % of them dead, while on the competition match v6 already calls
81-92 % of such frames dead (`scripts/py scripts/gap_dead_stats.py`, label-free on the match).
SkillCorner's seen-ball dead frames do resemble the target's tracked-ball stoppages (E5; nearest
player 1.8 m, ball 3.3 m/s, `scripts/tracked_dead_stats.py`), so that regime now has a judge.

Dead-side rule scalars, one at a time on three providers (`dead_sweep.py`, rule stage): the
dormant clues `w_quiet`, `w_edge`, `w_gap_dead` and a stronger `w_slow_far` move every provider
by under 0.006; `dead_bias` −0.8 gains +0.007 on SkillCorner and loses 0.010-0.012 on Metrica.
`w_exit` 1.5 gains +0.025 on SkillCorner at the rule stage, but with the residual in place the
gain is +0.002 and IDSSE noisy loses 0.008 (`exit_resid.py`). No rule change.

## 4. Candidate C1 `3p`: residual and verifier retrained with SkillCorner DEV — rejected

| check | result |
|---|---|
| SkillCorner DEV, two-fold cross-fitting | +0.0084 [+0.006, +0.011], 10/10 (phase +0.0097, 9/10) |
| Metrica + IDSSE nested (bar: no judge mean more than 0.005 lower) | Metrica −0.004 / −0.006, IDSSE **−0.011** / −0.005; 2/10 cells up |

It fails the non-inferiority bar on three of four judge means, so it never got a TEST look.
C2 (`drop`) is empty: no component is negative on DEV. C3 (`3p-rules`) was not run: its
precondition was C1 passing, and the sweeps above show no scalar that helps across providers.

**TEST, production v6, one look** (`sc_e1.py test`; ten matches main had never touched):
v5 config → v6 **+0.104 [+0.095, +0.114], 10/10** (0.424 → 0.528; phase labels +0.110, 0.457 →
0.567, 10/10). PathCRF step on phase labels +0.065 [+0.060, +0.070], 10/10; verifier +0.006
[+0.000, +0.012] (event) and +0.005 (phase).

## 5. What this does and does not show

- v6's gain over v5 holds on 20 matches of a third provider with real tracker noise and the
  target's ball-gap structure, at about the size measured on the synthetic-noise judges
  (+0.09..+0.12 there). Every component is non-negative; the verifier is the only one whose
  gain is not clearly above zero.
- It does not show the absolute score to expect: labels are provider proxies, and SkillCorner's
  dead-ball handling is worse than the target's for a reason specific to broadcast tracking.
- Adding a third provider to the training of the small learned parts trades Metrica/IDSSE
  accuracy for SkillCorner accuracy rather than improving both.
- Short possessions (section 2) stay the open loss; section 6 shows no scalar recovers them.

## 6. Follow-up: missed short possessions (2026-11-10)

`.orchestration/v7/short_diag.py`, v6 without verifier, true possessions under 4 s that are not
matched at IoU 0.5, by the evidence inside them:

| | touch votes for the right team (>= 2 frames) | kick by it within 3 m, no vote | ball unseen | team never within 3 m | near, no signal |
|---|---|---|---|---|---|
| Metrica clean / noisy | 67 / 41 % | 17 / 26 % | 1 / 14 % | 2 / 5 % | 13 / 15 % |
| IDSSE clean / noisy | 58 / 28 % | 29 / 19 % | 0 / 40 % | 2 / 4 % | 11 / 8 % |
| SkillCorner DEV | 30 % | 20 % | 12 % | 15 % | 23 % |

Where votes exist there are 10-15 of them on average and the summed team logit favours the
right team in 57-81 % of cases, so the observation is there and the decoder does not act on it.
Making it act does not help (`team_sweep.py`, three providers, PathCRF on the production alive
mask): touch-now 4 / 5 / 7 loses 0.004-0.04 on every Metrica and SkillCorner cell; the team
confidence cap 0.9 / 0.95 is inert (+-0.0005); `w_touch` 1.5 / 2 and team switch cost 12 / 14 / 20
trade one provider against another (largest: cost 20, SkillCorner +0.005, Metrica -0.004 /
-0.005); combinations of stronger touch evidence with a higher cost lose everywhere. Touch votes
are as frequent around false hand-overs as around true short ones, so weighting them harder
adds as many wrong possessions as right ones. The D016 values, several of them at their grid
edge, are a three-provider optimum. What is missing is precision of the touch observation
itself (which contact changed the ball's path), not its weight.
