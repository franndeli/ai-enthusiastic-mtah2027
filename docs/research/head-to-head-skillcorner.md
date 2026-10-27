# Head-to-head of the three team pipelines on 20 unseen matches (2026-10-27)

Question: which of our three pipelines generalises best to a match it has never seen, judged by the
official scorer? The public 5 minutes cannot answer it (17 possessions, noise band ~0.41, and not a
tuning set under the 23 Oct rules), so every pipeline ran, unchanged, on public matches that none
of them was trained on.

## Setup

- **Matches**: the 20 SkillCorner open-data matches (A-League 2024/25, MIT licence), broadcast
  tracking from a real tracker. The ball is missing on 40.7 % of graded frames (the ATD: 40.7 %),
  ~16 players per frame (ATD: 20.5). Labels as in `possession/skillcorner.py`: SkillCorner's team
  in possession carried through passes and loose balls, D from game interruptions; 293
  possessions and 51 % D per match.
- **Bundles** (`possession/bundle.py`): each match written as a complete challenge bundle (ATD
  text + metadata with 46 slots, 0-1 coordinates, speeds in m/s from positions; Smart Tagging
  with the two kick-offs only; frame,label ground truth), periods on the challenge's graded halves
  (2nd half from its kick-off). Graded frames the match does not reach are X (never scored).
- **Pipelines**, each with its own `run.py` and its own trained model, through
  `scripts/rehearsal.py`:
  - **v5**: this branch (`config/possession_v5.json`; trained on Metrica + DFL).
  - **Route A** (`feat/route-a-possession` @ baf223b): MLP + HMM, trained with its own
    `possession.train --fit-all` on the 7 DFL matches. The sandbox cannot download the raw DFL XML,
    so the tables were converted from our DFL files; its own leave-one-match-out run reproduces
    its documented numbers (Viterbi 0.5: 0.5596 vs 0.5590 documented; 1,002,651 frames, same
    spell counts), so the model is the same up to rounding. Smart Tagging weight 0 (its tag
    evidence is calibrated on the public 5 minutes at run time, and these matches have no tags).
  - **main** (Fran, @ 60d0bfc, "v4 last-touch ownership"): rules + Viterbi with
    `config/possession.json`, `--no-cache`.
  - **b50**: the official `starter/baseline.py`, window 50.
- **Score**: official `evaluation/possession_metrics.evaluate`; paired bootstrap over the 20
  matches (2,000 draws, 95 %).

## Results

| pipeline | spellF1 | macroF1 | possessions (truth 293) | alive/dead F1 | team acc. | D share (truth 51 %) |
|---|---|---|---|---|---|---|
| **v5** | **0.351** | **0.716** | 259 | **0.810** | **0.846** | 42.6 % |
| main (Fran) | 0.331 | 0.574 | 309 | 0.723 | 0.835 | 21.7 % |
| Route A (Javi) | 0.315 | 0.581 | 416 | 0.731 | 0.826 | 19.6 % |
| b50 (official baseline) | 0.233 | 0.655 | 496 | 0.793 | 0.795 | 31.4 % |

- v5 − main = **+0.021 [+0.011, +0.031]**, v5 ahead on 18 of 20 matches.
- v5 − Route A = **+0.036 [+0.027, +0.046]**, ahead on 20 of 20.
- v5 − b50 = +0.118 [+0.106, +0.130]. b50 scores 0.233 here and 0.2433 on the real match, and
  predicts 31.4 % D here against 32.3 % on the match: the bundles are about as hard as the match.

## Reading

- The gap is mostly **dead ball**. main and Route A call ~20 % of the frames D where the truth has
  51 %; their team calls are close to v5's (0.835 / 0.826 vs 0.846 on frames all agree are alive).
  A missed stoppage merges two possessions, which costs spellF1 as well as macroF1.
- Route A also **fragments**: 416 possessions for 293 (it learns its switching rate from DFL's
  per-touch `BallPossession`, ~440 changes per match); on the real match it predicts 371.
- Check for the other two pipelines, aggregate only: their D share on the full match. The
  challenge's labels come from corrected tracking ("D where the corrected tracking marks the ball
  dead", CHALLENGE.md), like DFL's `BallStatus`, which is dead on 41–52 % of in-half frames;
  EVALUATION.md calls dead ball "a large share of all frames". The public 5 minutes (19.6 % D) are
  not representative of that. A full-match D share near 20 % would point to the same loss.

## Caveats

- SkillCorner is not the ATD: broadcast camera, fewer players per frame, labels derived by us.
  What transfers is the ranking and the size of the dead-ball gap, not the absolute scores.
- Smart Tagging is not exercised (kick-offs only): main's tag-based dead-ball evidence and Route
  A's tag evidence may be worth something on the real match. They cannot explain a 30-point D gap.
- v5 was chosen partly with SkillCorner in view (v5 − v4 +0.011 there), so it has a small selection
  advantage on these matches; the margins above are larger than that.
- One run each; every pipeline is deterministic.

## Compliance notes found while reading the code (23 Oct rules)

RULES.md: the public 5 minutes "are not training data and not a tuning set". Route A's `run.py`
calibrates its Smart Tagging evidence on `evaluation/gt/..._public.csv` at run time
(`smart_tagging.game_log_evidence`), and its ball-dropout ratio (`train.DEAD_RATIO`) and the choice
of emission scale and tag weight were set on the public window. main's commit messages report
public-window scores next to its changes. Worth a team check before anything is submitted.

## Reproduce

```sh
python scripts/rehearsal.py SPEC.json --skillcorner <opendata>/data/matches \
    --sc-tracking <dir with <id>_tracking.jsonl> --sc-cache <cache dir> --work <work dir> --jobs 2
```

`config/rehearsal.example.json` is a SPEC for the four pipelines (one git worktree per branch;
Route A needs its trained `model.npz` in its work dir). ~45 s per match on 2 CPUs.
