# Improvement round I1–I5: fixing the team call (2026-10-26)

## Diagnosis that drove it

Oracle decomposition on DFL LOMO under ATD-like ball gaps (scale 0.5): model 0.324; perfect D
0.379; perfect team on alive frames 0.842. Team accuracy (both alive) was 0.838 with the ball
seen and 0.612 with it missing. The team call, not dead-ball detection, was the bottleneck.

## Units (each paired against the same harness: `possession.train.lomo`)

| unit | change | vit0.5 spellF1 (macroF1) | team seen / missing | matches improved |
| --- | --- | --- | --- | --- |
| reference | ball features + softmax | 0.3244 (0.773) | 0.838 / 0.612 | — |
| I1 `tactical.py` | ball-free formation contrasts, attack direction per half | 0.4751 (0.847) | 0.884 / 0.873 | 7/7 |
| I2 `trajectory.py` | last/next team in control (≤1.5 m, rel. speed <7 m/s, 0.1 s) | 0.4447 (0.805) | 0.898 / 0.623 | 7/7 |
| I3 `mlp.py` | 32-unit tanh MLP instead of softmax | 0.3669 (0.795) | 0.858 / 0.624 | 7/7 |
| I1 + I3 | | 0.5166 (0.877) | 0.895 / 0.867 | |
| **I5: I1 + I2 + I3** | `train.all_features` + `train_mlp` | **0.5590 (0.885)** | **0.912 / 0.870** | |

Integrated LOMO by scale: 0.2 0.5350, 0.5 0.5590, 1.0 0.5593, 2.0 0.5497, 4.0 0.5399
(predicted spells 250 / 295 / 327 / 357 / 384 of 437). Nearest-player smooth50: 0.2303.
Thresholds in I1/I2 were chosen on the same 7 folds (~20 variants each): mildly optimistic, but
neighbouring settings differ by 0.01–0.03 against gains of 0.12–0.15.

Transfer checks on the ATD (aggregates): I1 attack direction flips between halves and agrees with
the per-frame team-centroid sign on 98.5% / 96.6% of frames; I2 control fires on 34.1% of
seen-ball play frames (DFL 33.4%).

## I4 Smart Tagging (match only, no DFL equivalent)

Possession-phase codes agree with the public GT team on ~0.775 of covered alive frames; zone codes
(BALL IN FINAL THIRD/BOX, PLAYERS IN THE BOX) are uninformative (0.50); DEFENDING codes mirror the
attacking ones. Evidence = shrunk log-likelihood ratios, weight W chosen from a 2-fold
time split of the public window (held-out log-likelihood flat for w 0.5–0.75 -> W ≈ 0.25).
Calibrated on the public window: **its public scores below are in-sample**.

## Candidates on the match

| candidate | full-match possessions | public spellF1 [5–95%] | public macroF1 | team acc |
| --- | --- | --- | --- | --- |
| previous canonical (pre-round) | 387 | 0.400 [0.253, 0.547] | 0.744 | 0.871 |
| I5, scale 1.0 | 490 | 0.400 [0.263, 0.537] | 0.772 | 0.928 |
| I5, scale 0.5 | 412 | 0.443 [0.292, 0.595] | 0.760 | 0.920 |
| I5, scale 1.0, tags W 0.25 | 470 | 0.474 [0.326, 0.611] | 0.775 | 0.933 |
| **I5, scale 0.5, tags W 0.25 (canonical)** | **371** | **0.518 [0.353, 0.671]** | 0.770 | 0.934 |

The public window extrapolates to ~360 possessions. Scale 0.5 ties 1.0 on LOMO and matches the
count; tags move the count from 412 to 371. Model and CSV are MD5-stable across reruns.
