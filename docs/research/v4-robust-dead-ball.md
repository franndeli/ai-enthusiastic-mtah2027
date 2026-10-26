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

## Reproduce

```sh
git clone https://github.com/metrica-sports/sample-data ~/mtah-external/sample-data
python scripts/train_model.py ~/mtah-external/sample-data/data --version v4   # -> config/possession_v4.json
python scripts/train_model.py ~/mtah-external/sample-data/data               # -> config/possession_v3.json
python scripts/validate_external.py ~/mtah-external/sample-data/data --variants v4,v3 --noise all
git clone https://github.com/SkillCorner/opendata ~/mtah-external/skillcorner   # needs Git LFS
python scripts/validate_skillcorner.py ~/mtah-external/skillcorner/data/matches
python -m unittest discover -s possession -p 'test_*.py'
```
