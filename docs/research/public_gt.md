# Public ground truth: structure and metric sensitivity (T1)

Source: `scripts/py -m possession.public_gt_report` (public GT and committed baselines only, scored
with `evaluation/possession_metrics.py`). Random perturbations are means over seeds 0-9.

> **Sample-size warning.** This is 5 minutes, 17 possessions, 4 dead-ball stretches. One possession
> is ~6 spellF1 points. Use these numbers to understand the metric and to set sane priors, never to
> choose between models: the public window already misranks the baselines (EVALUATION.md).

## a. Structure

One graded span, video frames [11185, 18685) = 7500 frames. 21 segments: 17 spells, 4 D stretches.

Sequence: `A H A H A D H A H A H A H A D A H D A D H`

| | n | min s | median s | mean s | max s |
|---|---|---|---|---|---|
| H spells | 8 | 0.76 | 9.96 | 17.27 | 42.44 |
| A spells | 9 | 2.84 | 7.44 | 11.45 | 35.72 |
| all spells | 17 | 0.76 | 7.56 | 14.19 | 42.44 |
| D stretches | 4 | 5.52 | 16.40 | 14.69 | 20.44 |

- Short spells are real: 0.76 s, 1.32 s, 2.84 s, 3.24 s, 3.48 s, 3.60 s (6 of 17 under 4 s).
- Hand-overs: 12 direct H<->A; 4 via D (3 change team, 1 same team resumes after D).
- No D stretch is shorter than 5.5 s here; D is 20% of graded frames but 4 of 21 segments.

## b. Sensitivity (perturbations of the GT itself)

### (i) Uniform lag

| lag | 1 fr | 3 fr | 8 fr | 12 fr | 25 fr (1 s) | 50 fr (2 s) |
|---|---|---|---|---|---|---|
| spellF1 | 1.0000 | 1.0000 | 0.9529 | 0.9059 | 0.7765 | 0.5882 |
| macroF1 | 0.9973 | 0.9918 | 0.9782 | 0.9672 | 0.9331 | 0.8745 |

EVALUATION.md's lag table (0.9818 / 0.8933 / 0.6539 at 3 / 8 / 25 frames) is from a different
sample; on this window up to 3 frames is free and the short spells make lag bite from ~8 frames.

### (ii) Fragmentation: n injected flickers of length L inside spells

| flicker | L=1, n=1 | n=3 | n=5 | n=10 | n=17 | L=25, n=1 | n=5 | n=17 |
|---|---|---|---|---|---|---|---|---|
| opposite team: spellF1 | 0.937 | 0.836 | 0.740 | 0.584 | 0.438 | 0.919 | 0.704 | 0.400 |
| opposite team: macroF1 | 1.000 | 1.000 | 0.999 | 0.999 | 0.998 | 0.997 | 0.986 | 0.957 |
| D: spellF1 | 0.963 | 0.904 | 0.835 | 0.717 | 0.584 | 0.945 | 0.794 | 0.517 |

A flicker's cost is set by its **count, not its length**: a 1-frame team flicker costs ~6 points,
almost the same as a 1-second one, while macroF1 does not see it at all. A D flicker costs about
half (it splits a spell without adding an opposite-team spell).

### (iii) Centred 3-class majority smoothing of the perfect GT

| window | +/-25 | +/-50 | +/-125 |
|---|---|---|---|
| spellF1 | 0.9125 | 0.8267 | 0.5385 |
| macroF1 | 0.9979 | 0.9941 | 0.9636 |
| spells | 15 | 13 | 9 |

Smoothing is a ceiling: even on a perfect signal, +/-125 frames caps spellF1 at 0.54 (the
smooth125 baseline gets 0.42 on this window). Each erased short spell takes its neighbours'
match with it (H [A short] H becomes one long H: 17 -> 15 spells for one 0.76 s spell).

### (iv) X policies (X is forward-filled by the spell metric)

| policy | spellF1 | macroF1 | spells |
|---|---|---|---|
| X over middle 50% of every possession | 1.0000 | 0.7764 | 17 |
| X over +/-1 s around every boundary | 0.7333 | 0.9291 | 13 |
| X over every D stretch | 0.8121 | 0.6667 | 16 |

Mid-spell X is free on spellF1 and costs only macroF1 (the tie-break). X at boundaries is a 1 s lag
plus it swallows the shortest spells. X over D merges same-team spells across a stoppage.

### (v) D errors

| case | spellF1 | macroF1 |
|---|---|---|
| no D at all (D -> previous team) | 0.8121 | 0.5913 |
| D edges off by 3 / 12 / 25 fr (grown) | 1.0000 | 0.9959 / 0.9838 / 0.9668 |
| D edges off by 3 / 12 / 25 fr (shrunk) | 1.0000 | 0.9959 / 0.9832 / 0.9642 |
| D edges off by 50 fr (grown / shrunk) | 0.8941 / 0.9647 | 0.9357 / 0.9250 |

Without D, macroF1 collapses (the D class F1 goes to 0), but spellF1 only loses the one
same-team-resumes case plus IoU erosion. D boundaries within 1 s are free on spellF1.

### (vi) Merging short spells into the previous segment (a hard minimum duration)

| min duration | 1 s | 2 s | 3 s | 5 s | 8 s |
|---|---|---|---|---|---|
| spellF1 | 0.9125 | 0.8267 | 0.7429 | 0.5600 | 0.3455 |
| spells | 15 | 13 | 11 | 8 | 5 |

Merging all segment types (including D) is identical up to 5 s (no D is shorter than 5.5 s).

## c. Committed baselines (sanity, matches EVALUATION.md)

| baseline | spellF1 | macroF1 | spells |
|---|---|---|---|
| raw | 0.0378 | 0.5972 | 131 |
| smooth25 | 0.2868 | 0.6200 | 36 |
| smooth50 | 0.3256 | 0.6207 | 26 |
| smooth125 | 0.4242 | 0.5812 | 16 |

## Implications for a model + Viterbi pipeline

1. **Fragmentation is the dominant cost.** Every spurious switch costs ~6 points (team) or ~4
   (D) regardless of length; macroF1 cannot see it. Viterbi transition penalties should make a
   switch require sustained evidence, i.e. tuned so isolated sub-second disagreements never flip.
2. **But short spells exist.** Real spells as short as 0.76 s and 1.32 s appear in 5 minutes; a
   hard minimum duration of 1 s already costs ~9 points, 3 s ~26 points. Prefer a soft switch
   penalty (or a minimum duration no longer than ~0.5 s / 12 frames) over long hard minima.
3. **Lag is cheap up to ~0.1 s, then ~1 point per frame.** 8 frames -5, 25 frames -22 points.
   Centred (offline, two-sided) decoding is preferred over causal filtering, which lags.
4. **D boundaries are forgiving on spellF1** (+/-1 s free) but count for macroF1; D detection
   matters mostly via the tie-break and the same-team-resumes case. Prefer a confident D
   for long stoppages over precise D edges.
5. **X policy:** emit X only mid-spell under low confidence (free on spellF1, costs macroF1);
   never at a candidate boundary (that is a forced lag of the X length) nor over long stoppages.
6. Calibrate on these 5 minutes with `--bootstrap`; select models only on external LOMO (T4).
