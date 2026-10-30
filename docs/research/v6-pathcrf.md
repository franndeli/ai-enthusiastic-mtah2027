# v6: PathCRF as ball-free team evidence (2026-10-30)

Status: default of `run.py` since 30 Oct (`config/possession_v6.json`, decision D010), after the
validation on public data and the local check on the match below. v5 stays selectable with
`--model config/possession_v5.json`.

## Why

- The bottleneck of v5 is the team call while the ball is not seen: on DFL the oracle team call is
  worth +0.28 spellF1, the oracle dead ball +0.10 (v5 research notes).
- Tracking-only features of our own family were saturated: a stronger classifier, more formation
  contrasts and per-match self-training did not transfer (PLAN E5–E8). v5's boundaries are not
  late (E9): the loss is missed or invented changes of team.
- PathCRF (Kim et al., KDD 2026) is a different kind of evidence: a set-transformer + CRF trained
  to infer the possession path (which player controls the ball, which pass is in flight) from the
  movement of all 22 players, never looking at the ball.

## What v6 does

1. v5 as before: features → logistic models → Viterbi → H/A/D.
2. The ATD's 46 tracklet slots are re-tracked into 11 continuous tracks per team
   (`possession/retrack.py`: Hungarian assignment with a constant-velocity prediction; coasting
   stretches replaced by linear interpolation).
3. PathCRF's pretrained Dynamic Masked CRF (`third_party/pathcrf`, MPL-2.0, unmodified; trained on
   five Sportec DFL matches) runs inside the stretches v5 decoded as alive, on CPU.
4. Its probability that the last possessor is the home team, rounded to 3 decimals, becomes a
   log-odds (clipped at ±4) that is added to v5's team logit with weight 1; the same decoder runs
   again. Dead ball is unchanged.

Deterministic: fixed CPU thread count, deterministic algorithms, rounded probabilities. About 3
minutes per match on two CPU cores.

## When it helps: tracking quality

| id | data | players | v5 | v5 + PathCRF (w 1) |
|---|---|---|---|---|
| pc1 | DFL J03WR9 (held out by PathCRF), seed 10 | clean | 0.577 | 0.660 |
| pc2 | same | dropouts + identity swaps (~20.5 per frame) | 0.577 | 0.656 |
| pc3 | same | + 0.15 m frame jitter, smoothed over 1 s first | 0.577 | 0.628 (w 0.5: 0.633) |
| pc4 | same | + 0.15 m frame jitter, not smoothed | 0.577 | 0.583 |
| pc5 | Metrica games 1–3, seed 10 | dropouts + identity swaps | 0.550 | 0.594 |
| pc6 | SkillCorner, 4 matches | broadcast, 42 % of positions extrapolated | 0.368 | 0.347 |
| pc7 | same | only detected players (~13 of 22) | 0.368 | 0.342 |

A strong complement when nearly all players are tracked with smooth positions; harmful when a
third of them is missing or invented (broadcast tracking). So the decision rule was written down
before looking at the match (PLAN E10): objects ≥ 19 per frame and jitter ≤ 0.05 m → no
smoothing, weight 1; jitter up to 0.6 m → smoothing, weight 0.5; objects < 17 → no PathCRF.

The match (`scripts/tracking_report.py`, aggregates, Ángel's machine): 20.5 objects per graded
frame (p10/p50/p90 18/22/22), at least 11 per team on 44 % of frames, jitter (second difference)
median 0.0014 m (DFL clean 0.010 m, SkillCorner 0.002 m), teleports 0.05 %, 11.5 tracklets per
graded minute (median 35.5 s), duplicates 12 % (DFL clean 12 %). → no smoothing, weight 1.

## Validation of that exact setting

Simulation recalibrated to the match: no position jitter, dropouts and identity swaps as before;
v5 fold models that never saw the test match; seeds 10–12; controlled-possession labels (DFL) and
event-derived labels (Metrica).

| test set (held out by v5's fold model and by PathCRF) | copies | v5 | v6 | v6 − v5 [95 %] | possessions v5 → v6 (truth) |
|---|---|---|---|---|---|
| DFL J03WR9 + J03WQQ (PathCRF's test and validation matches) | 6 | 0.596 | **0.641** | **+0.044 [+0.028, +0.065]**, 6 of 6 better | 244 → 277 (276) |
| Metrica sample games 1–3 (another league) | 9 | 0.548 | **0.585** | **+0.037 [+0.007, +0.071]**, 8 of 9 better | 225 → 239 (381) |
| all | 15 | 0.568 | **0.607** | **+0.040 [+0.019, +0.061]**, 14 of 15 better | 233 → 255 |

Interval: bootstrap over matches (all seeds of a match together). Weight 0.5 would give +0.028
[+0.014, +0.042] (15 of 15); weight 1 was fixed beforehand. macroF1 also rises (0.859 → 0.866).
Dead ball is identical by construction (the dead-ball stage of the decoder does not read the team
logit). v6 predicts more possessions than v5, closer to the truth on DFL.

End to end, in the submission path: DFL J03WR9 written as a challenge bundle (ATD format, same
degradation, seed 10), `run.py` with v5 and with v6: valid files; spellF1 0.563 → 0.638,
possessions 235 → 263 (truth 267), D 43.1 % in both; v6 twice gives the same SHA-256; PathCRF
covered 100 % of the alive frames in ~4.5 min on two cores. (The shipped v5 was trained with this
match, which favours v5.)

SkillCorner is outside the regime: broadcast tracking with a third of the players missing or
extrapolated (pc6, pc7), where PathCRF hurts. The match is not like that (tracking report above).

## Local check on the match, before it becomes the default

Criteria, set beforehand: valid file; same SHA-256 on two runs; PathCRF covers ≥ 85 % of the alive
frames; D share identical to v5 (51.5 %); possessions above v5's 221 and below ~290 (validation:
+6 % to +14 %); public 5 minutes as a sanity check only.

Result (30 Oct, Ángel's machine, Windows, CPU torch 2.14.1; aggregates only): all pass.

| criterion | target | result |
|---|---|---|
| official validator | valid | valid |
| same SHA-256 on two runs | yes | yes: `c961f5f214912857933721c1661874499825e3b43ad91eade3f4f509d3fae120` |
| PathCRF coverage of the alive frames | ≥ 85 % | 100.0 % (101 s and 87 s) |
| D share of graded frames | 51.5 % (v5) | 51.5 % |
| possessions | 221 < n < ~290 | 247 (+12 %), mean 12.5 s, median 7.8 s |
| public 5 minutes, sanity only | H/A not inverted (macroF1 ≥ ~0.75) | macroF1 0.8587 (v5 0.8492), spellF1 0.7176 (v5 0.5733), 17 possessions (truth 17); D F1 identical (0.790) |

The public 5 minutes decide nothing (17 possessions; differences below ~0.33 spellF1 are noise and
the rules say they are not a tuning set); they only confirm that nothing is broken.

Reference environment: Windows, Python 3.11.9, numpy 2.4.6, pandas 3.0.6, torch 2.14.1+cpu,
torch_geometric 2.8.1, scipy 1.17.1 (pinned in `requirements-v6.txt`).

Still to do: the same SHA-256 on a second machine (Fran's), which is what the organisers will
check (RULES rule 3); `run.py` records the torch, torch_geometric and scipy versions in its
manifest to help if it differs.

## Credits and licence

PathCRF: H. Kim, K. Lee, S. Seo, S.-K. Ko, J. Yoon, C. Park. *PathCRF: Ball-Free Soccer Event
Detection via Possession Path Inference from Player Trajectories*. KDD 2026,
doi:10.1145/3770855.3818152. Code and weights: github.com/hyunsungkim-ds/pathcrf (commit 1b7c6f8),
Mozilla Public License 2.0; the vendored files are unmodified (`third_party/pathcrf/NOTICE.md`).
Its model was trained on the Sportec Open DFL dataset (Bassek et al. 2025, CC-BY 4.0), the same
external data v5 uses; nothing about the challenge match.
