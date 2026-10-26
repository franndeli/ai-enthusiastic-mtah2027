# T4 — leave-one-match-out validation on DFL (2026-10-26)

Command: `scripts/py -m possession.train` (aggregates only, ~100 s). Data: 7 DFL/Sportec matches
(Bassek et al. 2025, CC BY), 1,002,651 frames, labels from `BallStatus`/`BallPossession`.
Every held-out match gets the same seeded ball dropout as training (43% of frames, geometric
blocks of mean 1.5 s) to mimic the ATD. Scored with the official `possession_metrics.evaluate`.

Mean over the 7 held-out matches, spellF1 (macroF1):

| method | spellF1 | macroF1 | predicted spells (GT 370–511) |
| --- | --- | --- | --- |
| nearest player + smooth50 (`starter/baseline.py` logic) | 0.2319 | 0.394 | 556–648 |
| softmax, per-frame argmax | 0.1737 | 0.810 | 1212–1586 |
| softmax + Viterbi, emission_scale 0.2 | 0.3845 | 0.832 | 205–261 |
| **softmax + Viterbi, emission_scale 0.5** | **0.4048** | 0.829 | 254–340 |
| softmax + Viterbi, emission_scale 1.0 | 0.3909 | 0.825 | 318–422 |
| softmax + Viterbi, emission_scale 2.0 | 0.3564 | 0.821 | 412–501 |
| softmax + Viterbi, emission_scale 4.0 | 0.3156 | 0.818 | 539–667 |

Per match, Viterbi 0.5 beats the baseline on all 7 (gaps +0.12 to +0.19, noise band 0.04).
emission_scale 0.5 is best on 5 of 7 matches and an interior optimum of the grid.

## Reading

- The baseline scores 0.232 here and 0.2433 on the MTAH full match: DFL + dropout reproduces
  the baseline's level, which supports (does not prove) using it as the selection proxy.
- Temporal structure is the gain: the same posteriors go from 0.17 (argmax) to 0.40 (Viterbi).
- Selecting one scalar on the same folds is mild optimism; there is no nested split.

## Known gaps (to close in T9/T10 once the match data is available)

- **D share mismatch**: DFL `BallStatus=0` covers 41–52% of in-half frames; the public MTAH
  window has 19.6% D. Either the 5-minute window is unrepresentative or MTAH "dead" is narrower.
  Check the full-match predicted D share (aggregate) against both before trusting the D prior.
- Dropout is uniform in time; the ATD loses the ball disproportionately in crowded moments.
- No ATD-like position noise, slot fragmentation or duplicated players in training.
- Not yet measured on the MTAH public window (needs H1 data).
