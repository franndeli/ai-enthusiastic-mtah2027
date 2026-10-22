# Technical document — MTAH 2027 Phase 1, ball possession

Status: 2026-10-22, candidate v3 (rule evidence + Viterbi). Every number below is produced by a
command in this repository; nothing is hand-labelled or hand-corrected, and nothing is fitted on
the competition match (organisers' clarification of 2026-10-22, issues #2 and #8).

## 1. Approach

Per frame, the tactical tracking is turned into **log-evidence** for three states — home in
possession (`H`), away (`A`), dead ball (`D`) — and an offline **Viterbi decoder** picks the most
probable state sequence of each half under a fixed cost for every state change (6 nats for all
six transitions). Frames outside the graded halves are `X`; the graded frames before the
second-half kick-off (110,808–113,741) are `D`. Pipeline and signatures: `docs/PIPELINE.md`.

The evidence is two independent log-odds, combined so that team confidence never decides
alive-versus-dead (`possession/evidence.py`):

**Team log-odds (home vs away)**, capped at ±logit(0.75) per frame because neighbouring frames
are not independent votes:

1. *Sustained control* (`control_balance`): fraction of a centred 1.5 s window in which a player
   of each team is within 2 m of a controllable ball (ball speed ≤ 8 m/s; a faster ball is in
   flight and nobody controls it). A single touch or deflection barely moves it; a reception and
   carry does. Loose and flying balls give flat evidence, so the decoder keeps the previous owner
   (CHALLENGE.md: contested balls stay with the last team).
2. *Memory across gaps* (`w_gap`): where control is unclear (including the 41% of graded frames
   with no ball), the last and the next team seen in clear control, each decaying with a 5 s time
   constant. The pipeline is offline, so looking ahead to the re-acquisition is legitimate; a
   hand-over inside a gap is then placed where the two memories cross instead of at the gap edge.
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
6. *Smart Tagging*: a small constant inside tagged set-piece windows (`w_tag_dead` 0.5). Tags are
   evidence, never labels; their windows are coarse (end ~12 s after the real restart).

A missing ball is **missing evidence, not a dead ball**: it only flattens the team evidence; the
dead log-odds come from the players and from a projected exit.

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (players + ball) | all evidence |
| Smart Tagging (set pieces) | dead-ball prior inside tagged windows |
| Tactical homography, both videos | not used |
| Broadcast ATD / homography | **dropped** after a measured experiment (W7a): formation-based clock alignment accepted 31 of 253 broadcast runs and recovered 0.5% of the tactical ball gaps; the two balls disagreed by 2.6 m median where both were visible |
| IDSSE / Sportec Open DFL (Bassek et al. 2025, 7 Bundesliga matches, CC-BY 4.0; `scripts/fetch_idsse.py`) | external labelled data: fitting the four shape weights, and validating every design choice on ~3,000 labelled possessions, clean and with ATD-like noise |

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
| IDSSE leave-every-match-out, clean and noisy (`scripts/idsse_eval.py`) | the selection judge: 437 true possessions per run; noise = ball dropout with our gap statistics, jitter, slot fragmentation |

IDSSE's labels are finer than ours (D share 41–52%, spells mean 5.5–9 s versus 14 s in our public
window), so it is used for *relative* decisions, never to set our switching rate.

| Candidate | Public spellF1 [5–95%] | Public macroF1 | D P / R | Transplant mean / min | tc 5 / 10 / 20 s | IDSSE clean / noisy | Full-match possessions, mean, D share |
|---|---|---|---|---|---|---|---|
| `baseline_smooth50` (organisers) | 0.3256 | 0.6207 | — | — | — | — | — |
| v1 rules (control + memory + tags) | 0.4914 [0.34, 0.64] | 0.748 | 0.84 / 0.49 | 0.42 / 0.29 | 0.36 / 0.26 / 0.17 | 0.377 / 0.259 | 297, 14.7 s, 31% |
| v2 + ball-free shape (weights fitted on our match — withdrawn, non-compliant) | 0.5935 | 0.772 | 0.84 / 0.49 | 0.54 / 0.46 | 0.41 / 0.47 / 0.48 | 0.377 / 0.308 | 275, 15.8 s |
| v3 + dead-ball look-ahead and ball exit | 0.6133 [0.41, 0.79] | 0.824 | 0.87 / 0.65 | 0.55 / 0.46 | 0.42 / 0.49 / 0.48 | 0.405 / 0.332 | 260, 14.8 s, 39% |
| **v3c, shape weights refit on IDSSE (submitted)** | **0.6000 [0.39, 0.78]** | **0.827** | 0.87 / 0.64 | 0.54 / 0.43 | 0.40 / 0.42 / 0.48 | **0.414 / 0.345** | 264, 14.6 s, 39% |

Decoder switch costs were swept (2–12 nats, duration-aware variant up to 2 frames): inert on
IDSSE (±0.003) and within noise on the public window; 6/6/6 kept. Clues tried and rejected on
the judges: ball interpolation across gaps, centroid advance, a restart-team prior from tags,
longer or asymmetric memory, gap-length-as-dead, ball-at-rest and formation-based dead clues
(the last two cannot be validated on IDSSE, whose ball is frozen during dead time).

## 6. Limitations

- The ground truth comes from corrected tracking we never see; our dead-ball onset follows the
  players, theirs follows the referee/ball. Four public stoppages are all we could check against.
- Hand-overs inside long gaps (34.7% of graded frames sit in gaps > 10 s) are placed by shape
  and memory only; the transition-centred stress tests show spellF1 ~0.4–0.5 there.
- Short possessions (< 2 s) are under-detected (we predict 264 possessions; the public rate
  extrapolates to ~360).
- Public-window numbers have ±0.2 intervals; IDSSE semantics differ from ours in granularity.
- A learned evidence model trained on IDSSE (`possession/learned.py`) exists but is not in the
  submitted path; it is kept only if it beats the rules on all three judges.

## 7. Reproduce

    scripts/setup-env.sh                                   # conda env from environment.yml (pinned)
    scripts/py run.py --out evaluation/results/<team>.csv  # ~2 s after the first (cached) load
    scripts/py evaluation/validate_submission.py evaluation/results/<team>.csv

Data: unpack the bundle as in `data/README.md` or point `MTAH_DATA_DIR` at it; intermediates go
to `MTAH_WORK_DIR` (default `~/.mtah2027/work`). The pipeline has no randomness; two runs give
byte-identical files (`cmp`, checked). The shape-weight constants are regenerated by
`scripts/py scripts/fetch_idsse.py && scripts/py scripts/idsse_prepare.py && scripts/py
scripts/fit_shape_weights.py` (external data, deterministic); they are stored in
`config/possession.json` so reproduction does not need the download.
