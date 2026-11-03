---
id: technical-doc
type: architecture-design
status: draft
title: Technical document (submission deliverable)
parent: goal
references:
  - architecture
  - decisions
---

# Technical document — MTAH 2027 Phase 1, ball possession

Status: 2026-11-03, submitted **v6** = the rules below (all parameters re-selected on external
data, D016) plus four W6 components (§1b). Every
number below is produced by a command in this repository; nothing is hand-labelled or
hand-corrected, and nothing is fitted on the competition match (organisers' clarification of
2026-10-22, issues #2 and #8): learned parts are trained on external public matches only.

## 1. Approach

Per frame, the tactical tracking is turned into **log-evidence** for three states — home in
possession (`H`), away (`A`), dead ball (`D`) — and an offline **Viterbi decoder** picks the most
probable state sequence of each half under a fixed cost for every state change (16 nats between
the teams, 10 nats into and out of a stoppage). Frames outside the graded halves are `X`; the graded frames before the
second-half kick-off (110,808–113,741) are `D`. Pipeline and signatures: `docs/PIPELINE.md`.

The evidence is two independent log-odds, combined so that team confidence never decides
alive-versus-dead (`possession/evidence.py`). The constants quoted in the clue descriptions
below are the values each clue was designed with; the submitted values were all re-selected on
external data only by `scripts/tune_rules.py` (D016) and live in `config/possession.json` (for
example touch radius 1.25 m over 2 frames, touch-now 3 nats, team cap logit(0.85)).

**Team log-odds (home vs away)**, capped at ±logit(0.75) per frame because neighbouring frames
are not independent votes:

1. *Sustained control* (`control_balance`): fraction of a centred 1.5 s window in which a player
   of each team is within 2 m of a controllable ball (ball speed ≤ 8 m/s; a faster ball is in
   flight and nobody controls it). A single touch or deflection barely moves it; a reception and
   carry does. Loose and flying balls give flat evidence, so the decoder keeps the previous owner
   (CHALLENGE.md: contested balls stay with the last team).
2. *Last team to play the ball* (`w_touch`, v4): the public ground truth switches owner at the
   opponent's first touch and keeps him through flights, loose balls and clearances (it contains
   0.8 s and 1.3 s possessions). A touch vote is the nearest team while a visible ball is within
   1.5 m of one of its players, the other team at least 0.5 m farther, for at least 3 frames; the
   last vote persists causally, fading with a 20 s constant plus a 5 s constant of the time the
   ball has been unseen, so that in long gaps the ball-free shape takes over. The former
   symmetric gap memory (`w_gap`, hand-over at the middle of a gap) is switched off: on every
   external match the touch memory places hand-overs better.
   *Pass-by filter* (v5): a vote run during which the ball stays faster than 3 m/s and its
   velocity never changes (beyond a label-free jitter floor read off the match) is a ball flying
   past a player, not a touch, and is dropped (11 % of votes on the Metrica judge, at coin-flip
   precision). *Touch-now* (v5, `w_touch_now`): while a surviving vote is active, 2 nats are
   added beyond the cap below — a seen touch is the label by definition, the cap is for the
   proxies. This is what lets 1–4 s possessions (a third of all possessions) be decoded.
3. *Ball-free team shape* (`w_shape`): the possessing team is longer along its attacking axis
   and wider, the defending team runs more, and the whole block drifts towards the possessor's
   goal. Each team's attack direction is inferred per half from where it keeps its goalkeeper
   (label-free). The four home-minus-away differences are z-scored on the match (label-free
   normalisation) and combined with weights **fitted on the seven IDSSE matches only**
   (`scripts/fit_shape_weights.py`; constants recorded in `config/possession.json`).

**Dead log-odds**:

4. *Players walking*: smoothed mean player speed, judged by the minimum of "now" and "3 s ahead"
   so that a stoppage is called from its first frame rather than 3 s late (the dominant error
   before this clue: onsets were a median 3.2 s late on IDSSE).
5. *Ball beyond the lines*: share of the last second with the ball outside the pitch, and
   `ball_exited` — the last visible trajectory projected over a line and the ball not yet back.
6. *Smart Tagging*: **switched off in the submission** (`w_tag_dead` 0): no external data carries
   Smart Tagging, so its weight could only have been tuned on the provided game (D016). Designed as
   a small constant inside tagged set-piece windows. Tags are
   evidence, never labels; their windows are coarse (end ~12 s after the real restart).
7. *Ball travelling* (`w_moving`, v4): the share of a 1 s window with the ball seen faster than
   5 m/s lowers the dead log-odds; the ball stops before the players do and restarts before
   they accelerate, which fixed late restarts and early onsets on the external matches.
8. *Restart clues* (v5): a player standing more than 0.5 m beyond a line (`w_out_player`: a
   thrower or a ball retriever; needs no ball; 6–50 times likelier on dead frames on every
   external match) and a seen ball slower than 2 m/s with nobody within 2.5 m for most of a
   second (`w_slow_far`: a ball waiting for a restart; alive false-positive rate ≤ 1.4 % on
   every judge).

A missing ball is **missing evidence, not a dead ball**: it only flattens the team evidence; the
dead log-odds come from the players and from a projected exit.

### 1b. v6 additions (W6, D015)

First the external judges were made to look like the target, using label-free statistics only:
the target's ball track is smooth (frame-to-frame second difference 0.9 cm), its players drop out
(20.5 seen on average, 14 at the 5th percentile) and its ball sits ~1 m from the nearest player
where the Metrica sample's sits on the player. The noisy judges now use that noise
(`external.inject_matched_noise`); the old iid 0.3 m ball jitter does not exist on the target.
On top of the v5 logits, v6 adds:

9. *Fresh-exit onset* (`w_fresh_exit` 2.0): the first 5 s after the ball leaves the pitch, if it
   stays out for 2 s. v5's dead clues stayed flat for 3-5 s after an exit; this clue is 20-150
   times likelier on the first seconds of a stoppage than in play on both external judges.
10. *Ball-free dead residual* (`possession/dead_residual.py`): a gradient-boosted correction of
    the dead logit (clipped to +-1 nat) from the rule logit and player-only context (team speeds,
    spreads, separation, slow and out-of-pitch players, 0.5-4 s windows). A version that also read
    the ball learned provider artefacts and failed when trained on one provider and tested on the
    other; this one transfers (19/20 cells).
11. *PathCRF team evidence* (`possession/pathcrf.py`; Kim et al., KDD 2026; vendored, MPL-2.0,
    integrated first by a teammate): a pretrained set-transformer + CRF infers the ball possessor
    from the 22 players' movement alone. Tracklets are re-tracked into 11 tracks per team, PathCRF
    runs inside the stretches v5 decodes as alive, and half its home log-odds (clipped at +-4) is
    added to the team logit. Its value is where the ball evidence is unclear (57 % of the
    target's alive frames, like the matched-noise judges).
12. *Spell-edit verifier* (`possession/verifier.py`): the same evidence is decoded with every
    switch cost scaled by 0.35-2; each stretch where a scaled path differs is a proposed edit (an
    extra short possession, a merge, a moved stoppage). A gradient-boosted regressor predicts each
    edit's effect on spellF1 from spell-level features and accepted edits are applied greedily.
    It only chooses among paths the decoder itself produces.

The two learned models are trained by `scripts/train_v6.py` on the Metrica sample and IDSSE
matches (clean and matched noise) and stored in `models/` with their SHA-256 in the config.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players + ball) | all evidence |
| Smart Tagging (set pieces) | implemented as a dead-ball prior, **off** in the submission: no external tagged data to tune it on (D016) |
| Tactical homography, both videos | not used (a teammate's YOLO pilot on the tactical video failed its agreement test: top detection within 2 m of the tracked ball on only 53-59 % of frames) |
| PathCRF pretrained model (Kim et al., KDD 2026, github.com/hyunsungkim-ds/pathcrf, MPL-2.0; trained on five Sportec Open DFL matches) | ball-free team evidence (§1b); evaluated only on matches it never trained on |
| Broadcast ATD / homography | **dropped** after a measured experiment (W7a): formation-based clock alignment accepted 31 of 253 broadcast runs and recovered 0.5% of the tactical ball gaps; the two balls disagreed by 2.6 m median where both were visible |
| IDSSE / Sportec Open DFL (Bassek et al. 2025, 7 Bundesliga matches, CC-BY 4.0; `scripts/fetch_idsse.py`) | external labelled data: fitting the four shape weights, and validating every design choice on ~3,000 labelled possessions, clean and with ATD-like noise |
| Metrica Sports sample data (3 anonymised matches, github.com/metrica-sports/sample-data; `scripts/fetch_metrica.py`) | primary external judge since v4: same provider and formats as the competition data, 25 fps, 0–1 coordinates, ball missing 32–41% natively; possession labels derived from the synchronised events with last-touch semantics (`possession/metrica.py`), ~1,000 possessions |

## 3. Handling missing and noisy tracking

Measured on the full match (`scripts/audit_data.py`): ball visible on 59.3% of graded frames;
150 missing runs, median 10.7 s, p90 34 s, max 221 s; 34.7% of graded frames lie in gaps longer
than 10 s; tracked objects 21.1 per frame on average (p5 14). The public window has the ball 88%
of the time, so it understates the problem: on the public window, keeping the last seen team is
right on 94% of missing frames, and only 1 of its 20 transitions falls inside a gap.

Design consequences: no interpolation of the ball across gaps (tested, rejected); team evidence in
gaps comes from the memory and the ball-free shape clues; slot identity is never used (46
anonymous, fragmented slots), every feature is a per-team aggregate; speeds use the tracker's
player speed and a ball speed re-derived from positions (the tracker's ball speed is mostly
missing).

Robustness is measured with two stress tests on the public window (`scripts/stress_test.py`):
*transplant* hides the ball in runs drawn from the real full-match gap distribution until 41% is
missing (5 seeds), *transition-centred* hides it for 5/10/20 s around every true hand-over.

## 4. Possession rules implemented

- A change of possession needs sustained control by the other team, not a touch; a deflection,
  a header without control or a loose ball keep the owner (flat evidence + switch cost).
- Dead ball starts when the players stop (look-ahead) or the ball leaves the pitch, and ends when
  play restarts; the restart team is whoever shows control first. Set-piece ownership is not
  asked for, so no rule assigns it.
- Possessions never cross halftime; each half is decoded on its own.
- `X` is written only where the ground truth cannot score (outside the halves); inside, the
  decoder always answers (an `X` would only repeat the previous label on the ranking metric and
  cost macroF1).

## 5. Validation and results

Three judges, none of them the private ground truth:

| Judge | Role |
|---|---|
| Public 5 minutes (17 possessions), 1000-draw bootstrap | pipeline correctness; too small to select between variants (the organisers' own baselines flip order between public and full match) |
| Stress tests on the public window | gap robustness |
| IDSSE clean and noisy (`scripts/idsse_eval.py`) | retrospective development comparison, not held-out evaluation: production shape weights were fitted on all seven matches; noise = ball dropout with our gap statistics, jitter, slot fragmentation |

Validation audit (V1): the historical IDSSE numbers below are not proof of generalization.
`scripts/idsse_eval.py` does not refit the pooled shape weights per held-out match. The separate
learned experiment excludes evaluation matches from classifier fitting, but its shared calibration
temperature uses their labels (`fit_with_calibration`), so its reported LOMO scores also have
selection leakage. That experimental model remains outside production. Strict comparisons must
exclude confirmation labels from every fitted component, including calibration. Reused matches
and the 17-possession public window cannot establish that v3c is the best possible model.

IDSSE's labels are finer than ours (D share 41–52%, spells mean 5.5–9 s versus 14 s in our public
window), so it is used for *relative* decisions, never to set our switching rate.

| Candidate | Public spellF1 [5–95%] | Public macroF1 | D P / R | Transplant mean / min | tc 5 / 10 / 20 s | IDSSE clean / noisy | Full-match possessions, mean, D share |
|---|---|---|---|---|---|---|---|
| `baseline_smooth50` (organisers) | 0.3256 | 0.6207 | — | — | — | — | — |
| v1 rules (control + memory + tags) | 0.4914 [0.34, 0.64] | 0.748 | 0.84 / 0.49 | 0.42 / 0.29 | 0.36 / 0.26 / 0.17 | 0.377 / 0.259 | 297, 14.7 s, 31% |
| v2 + ball-free shape (weights fitted on our match — withdrawn, non-compliant) | 0.5935 | 0.772 | 0.84 / 0.49 | 0.54 / 0.46 | 0.41 / 0.47 / 0.48 | 0.377 / 0.308 | 275, 15.8 s |
| v3 + dead-ball look-ahead and ball exit | 0.6133 [0.41, 0.79] | 0.824 | 0.87 / 0.65 | 0.55 / 0.46 | 0.42 / 0.49 / 0.48 | 0.405 / 0.332 | 260, 14.8 s, 39% |
| v3c, shape weights refit on IDSSE | 0.6000 [0.39, 0.78] | 0.827 | 0.87 / 0.64 | 0.54 / 0.43 | 0.40 / 0.42 / 0.48 | 0.414 / 0.345 | 264, 14.6 s, 39% |
| v4, last-touch ownership + ball travel | 0.5250 [0.35, 0.70] | 0.811 | 0.95 / 0.60 | 0.49 / 0.40 | 0.34 / 0.42 / 0.40 | 0.508 / 0.397 | 314, 12.7 s, 37% |
| v5, pass-by filter + touch-now + restart clues | 0.5333 [0.36, 0.69] | 0.823 | — | 0.50 / 0.40 | 0.32 / 0.42 / 0.40 | 0.576 / 0.435 | 305, 12.7 s |
| **v6, W6 stack, rules re-tuned externally (submitted)** | **0.6235 [0.46, 0.78]** | **0.830** | — | §5 stress below | §5 stress below | see below | 310, 41.5 % D |

v6 against v5 on the target-matched judges, fresh noise seeds, every learned part trained
without the evaluated match (nested; `scripts/eval_v6_external.py`), on the five matches PathCRF
never trained on: Metrica clean/noisy 0.605 / 0.495 -> 0.654 / 0.585 (+0.049 / +0.091), IDSSE
0.592 / 0.469 -> 0.660 / 0.590 (+0.068 / +0.121); 10 of 10 match cells up, smallest +0.036.
The rule parameters were tuned on the clean and seed 0/1 copies of these same external matches,
so the clean cells are in-sample for that step; the 3 SkillCorner matches never used anywhere
give rule stage 0.470 -> 0.498 (weak, phase-derived labels). The public window was
looked at only after selection (`docs/PLAN.md` W6).

Stress tests on the public window (`scripts/stress_test.py --seeds 5`, robustness checks, not
selectors), v5 -> v6: transplant (41 % ball missing, 5 seeds) mean 0.493 -> 0.644, min 0.400 ->
0.579; transition-centred 5 / 10 / 20 s 0.317 / 0.415 / 0.400 -> 0.678 / 0.752 / 0.813. Hiding
the ball around hand-overs no longer hurts: PathCRF reads the players, not the ball.

v4 and v5 are also scored on the Metrica sample matches (clean / noisy spellF1): v3c
0.461 / 0.439, v4 0.563 / 0.518, v5 **0.605 / 0.570**; at each step every one of the ten
external matches improves, clean and noisy. v4's public-window decrease is inside its own
interval and is entirely dead-ball handling: with the true D frames supplied as a diagnostic
both v3c and v4 score 0.600. The organisers' warning that the public window ranks even their
own baselines wrongly, and the size of the external sample, decided the adoptions
(`docs/DECISIONS.md` D011 and D014, `docs/research/last_touch_ownership.md`,
`docs/research/touch_precision.md`). Oracle experiments on the Metrica judge (true D frames, or
true team labels, injected into v4's evidence) give 0.657 and 0.810 against 0.563: team
ownership is the larger remaining loss externally, stoppages on the public window.

Decoder switch costs were swept (2–12 nats, duration-aware variant up to 2 frames): inert on
IDSSE (±0.003) and within noise on the public window; 6/6/6 kept for v3c. v4 uses 16/8/8 (D011). Clues tried and rejected on
the judges: ball interpolation across gaps, centroid advance, a restart-team prior from tags,
longer or asymmetric memory, gap-length-as-dead, ball-at-rest and formation-based dead clues
(the last two cannot be validated on IDSSE, whose ball is frozen during dead time).

A subsequent factorized-learning review (V2, `docs/research/factorized_generalization.md`)
repaired the fitting boundaries in an isolated evaluator. Fixed 2 s ownership-label coarsening
and a team-only learner improved reserved IDSSE noisy spellF1 0.38329→0.43101, but target
spellF1 fell 0.60000→0.55484 and all target stress scores declined. It was **not adopted**.
A frozen comparison on three newly acquired SkillCorner matches yielded only +0.01166 mean
spellF1 against explicitly weak provider-derived phase labels, not independent corrected truth.
These experiments strengthen the transfer warning; they do not change the production method.

## 6. Limitations

- The ground truth comes from corrected tracking we never see; our dead-ball onset follows the
  players, theirs follows the referee/ball. Four public stoppages are all we could check against.
- Hand-overs inside long gaps (34.7% of graded frames sit in gaps > 10 s) are placed by shape
  and memory only; the transition-centred stress tests show spellF1 ~0.4–0.5 there.
- Possessions under 2 s are still mostly missed (external recall ~0.1–0.2 at IoU 0.5); 2–4 s
  ones are found about half the time. 77 % of true hand-overs have a touch vote of the new team
  within a second; the rest are contested balls where no team is unambiguously nearest.
- The pass-by filter relies on ball velocity; under the synthetic 0.3 m jitter its threshold
  rises with the match's own jitter floor and it is nearly inactive, so its gain is measured
  on clean external tracking. The automatic tracking's real ball jitter is not known.
- Stoppages in which the automatic tracking keeps seeing a slow ball with players around it
  (the public window's long D stretches) are under-detected, and v4 then splits them into short
  spurious spells; no external data reproduces that regime, so it could not be fixed without
  tuning on the match.
- Public-window numbers have ±0.2 intervals; IDSSE semantics differ from ours in granularity.
- A learned evidence model trained on IDSSE (`possession/learned.py`) exists but is not in the
  submitted path; it is kept only if it beats the rules on all three judges.

## 7. Reproduce

    scripts/setup-env.sh                                   # conda env + pinned CPU torch
    scripts/py run.py --out evaluation/results/<team>.csv  # ~90 s (PathCRF ~80 s on 4 threads)
    scripts/py evaluation/validate_submission.py evaluation/results/<team>.csv

Data: unpack the bundle as in `data/README.md` or point `MTAH_DATA_DIR` at it; intermediates go
to `MTAH_WORK_DIR` (default `~/.mtah2027/work`). The pipeline has no randomness; two runs give
byte-identical files (`cmp`, checked). The shape-weight constants are regenerated by
`scripts/py scripts/fetch_idsse.py && scripts/py scripts/idsse_prepare.py && scripts/py
scripts/fit_shape_weights.py` (external data, deterministic); they are stored in
`config/possession.json` so reproduction does not need the download. The v6 models are
regenerated by `scripts/py scripts/train_v6.py` (external data only, ~18 min, deterministic;
prints the SHA-256 recorded in the config). v5 stays reproducible with
`--config config/possession_v5.json`. The same file (sha256 `263bfddea0d0eeb7`) is produced on macOS
arm64 (conda env) and on Linux x86_64 (pip-only `python:3.11.9-slim` container,
`pip install -r requirements-pipeline.txt`; PathCRF's vendored code also needs matplotlib).
