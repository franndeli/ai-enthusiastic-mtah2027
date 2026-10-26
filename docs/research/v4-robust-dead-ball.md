# v4: a dead-ball model that survives a change of tracker

Status: implemented, validated on public external data, default in `run.py`. Not yet run on the
match: that run is local-only (E3 in PLAN.md). All numbers below are aggregates on public data
about other games, never on the match.

## Why

The first local run of v3 on the match (aggregates only, 2026-10-25) labelled **61 % of graded
frames D**, with 220 possessions. Other teams publish 35–47 % D and 237–320 possessions for the
same match, and a football match has roughly 55–60 min of live play. v3 over-calls stoppages on
the match.

Mechanism, reproduced on the external benchmark: v3's dead-ball model learned "players are slow"
from 0.24 s speeds on our degraded sample games, where per-frame position noise inflates every
speed by ~1 m/s. A tracker with smoother positions (as the ATD may be) gives lower speeds
everywhere, so live play looks like a stoppage. With smooth player positions at test time
(`smooth_players`), v3 over-calls D by **+15 pp** and loses 0.08 spellF1.

## What changed (`config/possession_v4.json`)

1. **Dead-ball features that do not depend on the tracker's noise level**
   (`features._dead_v4_features`):
   - player speeds from a 1 s displacement (position noise of ~15 cm per frame adds ~0.3 m/s
     instead of ~1 m/s);
   - the median speed divided by its median over the same half (`d4_rel_*`), so a tracker,
     league or camera that scales every speed by a constant does not move the dead-ball call;
   - shares of slow (< 1.5 m/s) and fast (> 4 m/s) players;
   - windows of ±1, ±4 and ±15 s, plus the 3 s before and the 3 s after each frame (the restart
     and the whistle are asymmetric in time);
   - ball evidence kept from v3 (out of bounds, ball speed, stillness, centre spot, visibility and
     gap features) plus the share of fast ball over 1 s.
2. **Domain randomisation.** Each degraded training copy of a sample game uses a different random
   noise model (`external.random_noise(seed)`: ball missing in play 15–40 % and in stoppages
   45–85 %, gap median 5–15 s, ball noise 0.1–0.6 m, player noise 0–0.3 m, independent or
   drifting over 0.5–2 s, dropouts 2–12 %). The model cannot rely on one guess of the ATD's noise.
3. **Team evidence: the last team in control, carried forward** with decays of 2, 6 and 20 s
   (`t_carry_*`), which follows CHALLENGE.md: a loose ball stays with the last team.

Unchanged: the two logistic regressions, the two-stage Viterbi, the switching costs (40 / 8, again
chosen by the in-sample grid), pre-kickoff handling, output and validation. v3 stays available
(`--model config/possession_v3.json`) and regenerates byte for byte.

## Evidence

### 1. Metrica sample games, leave one game out (`scripts/validate_external.py --noise all`)

Fit on two games (seeds 0–1), score the third (seeds 10–12) under each test noise; 3 games × 3
seeds per cell. GT has 381 possessions per game.

| test noise | b50 | v3 | **v4** | v3 D error | v4 D error |
|---|---|---|---|---|---|
| default | 0.247 | 0.544 | **0.557** | +2.9 pp | +2.3 pp |
| more ball missing in play | 0.210 | 0.515 | **0.530** | +4.4 pp | +3.7 pp |
| ball often visible in stoppages | 0.217 | 0.513 | **0.522** | +3.5 pp | +3.0 pp |
| longer gaps (median 15 s) | 0.258 | 0.549 | **0.561** | +2.9 pp | +2.1 pp |
| much noisier positions | 0.197 | 0.361 | **0.472** | −38.1 pp | −7.3 pp |
| smooth players | 0.250 | 0.468 | **0.559** | +15.0 pp | +3.4 pp |
| drifting players | 0.244 | 0.539 | **0.547** | +4.6 pp | −0.2 pp |
| **mean / worst** | 0.232 / 0.197 | 0.499 / 0.361 | **0.536 / 0.472** | | |

v4 is ahead in every scenario; the gap is small where our default noise guess is right and large
where it is wrong, which is the point. Paired v4 − v3 (`--ref v3`): default +0.013
[+0.006, +0.020]; smooth players +0.091 [+0.080, +0.102]; much noisier positions +0.110
[+0.099, +0.120].

### 2. SkillCorner open data: a real tracker (`scripts/validate_skillcorner.py`)

20 A-League 2024/25 matches of broadcast tracking (github.com/SkillCorner/opendata, MIT), 10 fps
upsampled to 25 fps. Labels from SkillCorner's possession and game-interruption data
(`possession/skillcorner.py`). The shipped configs, unchanged, as `run.py` applies them:

| model | spellF1 | macroF1 | possessions (gt 294) | D error |
|---|---|---|---|---|
| b50 | 0.233 | 0.650 | 498 | −19.9 pp |
| v3 | 0.303 | 0.690 | 253 | +5.7 pp |
| **v4** | **0.341** | 0.681 | 290 | −13.5 pp |

Paired over GT possessions: v4 − v3 = **+0.036 [+0.030, +0.042]**; v4 − b50 = +0.108
[+0.102, +0.115]. Absolute values are lower than on Metrica because the labels follow a different
convention and the tracker is a broadcast one; the paired difference is the claim.

Also tried and rejected: adding SkillCorner matches to training. It lowers the Metrica held-out
score (different label conventions), so SkillCorner is kept for validation only.

### 3. Dress rehearsal (`scripts/make_fixture.py`)

Full-size ATD-format bundle built from sample game 1, models retrained without game 1
(`MTAH_EXCLUDE_GAME=1`), scored by the official `run_evaluation.py`: v4 0.581, v3 0.546, b50
0.287. Same SHA-256 on two runs.

## The missing ball (DATA.md: absent on 41 % of graded frames)

How it is handled, layer by layer:

- **Benchmark.** The degradation hides the ball on 40.7 % of graded frames (the match: 40.7 %,
  i.e. 59.3 % seen, measured by our reader and stated in DATA.md), in bursts weighted towards
  crowded situations during play and mostly inside stoppages; 80 % of the missing frames fall in
  gaps of 10 s or more (other teams publish 86 % for the match). With it, the official baselines
  score what they score on the match (smooth50 0.247 vs 0.2433, smooth125 0.222 vs 0.2242). v4
  trains on 15–40 % missing in play, 45–85 % in stoppages and gap medians of 5–15 s.
- **Features.** "Ball not seen" is kept apart from "ball dead": the dead-ball model reads the
  players' movement, which needs no ball, plus gap length and whether a gap starts or ends near a
  line. Team evidence in gaps comes from team shape (2 s and 6 s) and, in v4, from the last team in
  control carried forward. Gaps up to 0.4 s are interpolated, impossible ball jumps dropped.
- **Decoder.** Without evidence the Viterbi keeps the current team (CHALLENGE.md: a loose ball stays
  with the last team). No `X` on graded frames: spellF1 forward-fills it anyway and macroF1 counts
  it as a miss.

Where it still costs us (external benchmark, default noise, leave one game out, 3 games × 3
seeds; `scripts/gap_benchmark.py`):

| graded frames | share | truth H / A / D | v4 H / A / D | v4 frame accuracy | team changes per gap, truth / v4 |
|---|---|---|---|---|---|
| ball seen | 59.3 % | 41 / 39 / 20 % | 41 / 39 / 20 % | 88.5 % | — |
| gap < 1 s | 0.2 % | 45 / 40 / 15 % | 35 / 35 / 30 % | 48.9 % | 0.01 / 0.04 |
| gap 1–10 s | 7.8 % | 27 / 21 / 52 % | 22 / 17 / 61 % | 70.8 % | 0.19 / 0.12 |
| gap ≥ 10 s | 32.6 % | 16 / 16 / 68 % | 13 / 13 / 74 % | 86.4 % | 0.55 / 0.24 |

Long gaps are mostly stoppages and v4 gets most of their frames right, but it detects fewer than
half of the possession changes that happen while the ball is missing, and over-calls D inside gaps
by 6–9 pp. Each missed change merges two possessions, which costs spellF1 twice. With more of the
ball missing in play (`more_missing_in_play`), truth has 0.81 changes per long gap and v4 0.38.

On the match, `scripts/gap_report.py` prints the same split (without the truth columns, which
only exist on the benchmark): missing share, gap lengths, the public window's own missing share,
and each submission's label mix and team changes inside gaps.

## What to check on the match (local, aggregates only)

- D share: v4 should be well below v3's 61 %, inside or near the 35–47 % others report.
- Possessions: more than v3's 220.
- Public 5 minutes: a sanity check only (±0.33 noise, 17 possessions; RULES.md: not a tuning set).

## Where the remaining error is

Oracle swaps on SkillCorner: perfect dead ball → 0.48, perfect team → 0.75 (v4 0.34). Team
evidence remains the bottleneck, as on Metrica. v4 also still predicts ~250 possessions per
Metrica game against 381, so short possessions are missed.

Tried and not kept: separate weights for the team-shape and carry features while the ball is not
seen (gated copies). Metrica +0.002 to +0.006, SkillCorner +0.001 [−0.001, +0.003]: not worth
the extra coefficients.

Next candidates: more labelled games with per-frame possession (Sportec Open DFL, PLAN X1), a
team model with interactions (e.g. gradient boosting kept antisymmetric by training on team-swapped
copies), and short-possession recall.

## On the match (local run, 2026-10-26, aggregates only)

- Valid; the same SHA-256 on two runs; v3 regenerated identical to its 25 Oct output.
- 223 possessions, mean 14.1 s, median 8.9 s, 3 under 1 s; D 50.4 % of graded frames (v3: 61.2 %),
  i.e. ≈ 52 min of play out of ≈ 106 graded minutes (v3 ≈ 41 min).
- Missing ball (`gap_report.py`): 40.7 % of graded frames; of the missing, 85.2 % in 79 gaps of
  10 s or more. Public 5 minutes: only 11.8 % missing. D share b50 / v3 / v4 with the ball seen
  0 / 37.9 / 23.0 %, in 1–10 s gaps 43.5 / 79.5 / 68.5 %, in gaps ≥ 10 s 85.8 / 97.7 / 94.2 %.
- Public 5 minutes, sanity only (the scorer's own noise band is ~0.41): spellF1 0.5733, macroF1
  0.8669, 13 possessions (gt 17). v3 had labelled 820 home-possession frames D there, v4 91.

## Tried after v4 (not kept)

- **Cheaper team switches at touches.** A per-frame switching cost (touch / ball seen / ball
  missing, chosen on the training games): Metrica −0.001 to +0.001 (CIs include 0), SkillCorner
  +0.0006 [−0.0013, +0.0022]. Not kept; the per-frame cost support stays in `decode.py` (tested)
  for tag-based change points (PLAN X2).
- **Which possessions are missed** (`gap_benchmark.py`): on the Metrica labels 53 % of the true
  possessions last under 5 s, and v4 recalls 7 % of those under 2 s and 34 % of those of 2–5 s
  (64 %, 78.5 % and 84 % for 5–10 s, 10–20 s and longer). Most misses are the other team's brief
  touches.
- **Benchmark labels closer to "controlled possession".** Our event-derived labels give 361–402
  possessions per game (mean 8.8–9.8 s); the public window has 17 of mean 14.2 s (DATA.md), and
  CHALLENGE.md does not count touches without control. A variant where the other team's run of
  events only changes possession if it contains a pass, shot or carry, or lasts ≥ 2 s, gives
  283–386 per game. Scored against it, v4 still beats v3 (+0.010 [+0.003, +0.017]; smooth players
  +0.100), and retraining v4 on it does not help (−0.004 [−0.008, −0.001]). So v4 stands, and the
  missed short possessions matter less for the challenge than the Metrica labels suggest. Labels
  derived from corrected tracking, like the challenge's, are the better yardstick: Sportec DFL
  (PLAN X1, `scripts/fetch_dfl.py`). Scratch experiments, not committed.

## Reproduce

```sh
git clone https://github.com/metrica-sports/sample-data ~/mtah-external/sample-data
python scripts/train_model.py ~/mtah-external/sample-data/data --version v4   # -> config/possession_v4.json
python scripts/train_model.py ~/mtah-external/sample-data/data               # -> config/possession_v3.json
python scripts/validate_external.py ~/mtah-external/sample-data/data --variants v4,v3 --noise all
git clone https://github.com/SkillCorner/opendata ~/mtah-external/skillcorner   # needs Git LFS
python scripts/validate_skillcorner.py ~/mtah-external/skillcorner/data/matches
python scripts/gap_benchmark.py ~/mtah-external/sample-data/data
python -m unittest discover -s possession -p 'test_*.py'
# on the match, local only, aggregates only
python scripts/gap_report.py evaluation/results/v4.csv evaluation/results/v3.csv
python scripts/tag_report.py --submission evaluation/results/v4.csv
```
