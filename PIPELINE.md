# Our possession pipeline — what we built and how to use it

Short version for the team. Details and every number: `docs/research/` (v3, v4, v5 notes),
`docs/PLAN.md` and `docs/DECISIONS.md`.

## In one minute

- For each of the 199,319 frames of the tactical video we output `H`, `A` or `D` (and `X` outside
  the graded halves), from the automatic tracking only.
- The current version is **v5**. It is the default of `run.py`.
- We **never tune on the public 5 minutes**. We test on **30 full matches from other teams** that do
  have labels, with their tracking made as bad as ours.
- On the DFL matches v5 scores **0.60 spellF1, against 0.25 for the official baseline**
  (SkillCorner: 0.35 against 0.23).

## Run it

```sh
python run.py --out evaluation/results/v5.csv                                        # v5 (default)
python run.py --out evaluation/results/v4.csv --model config/possession_v4.json      # older versions
python -m unittest discover -s possession -p "test_*.py"                             # 13 tests
```

- The match data is read from `~/.mtah2027/data/tactical/` (or set `MTAH_DATA_DIR` to the folder
  that contains `tactical/`).
- It prints only aggregates (possessions, % dead ball), runs the official `validate_submission.py`
  on the exact file and gives the same SHA-256 on every run. No training at run time.
- Windows (PowerShell): `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`, then
  `.\.venv\Scripts\Activate.ps1`.

## How it works

1. **Read the tracking** (`possession/tracking.py`): players and ball per frame; the ball is missing
   on 41 % of graded frames.
2. **Evidence per frame** (`possession/features.py`):
   - *Who has the ball*: nearest players of each team, who controls it, dribbles, receptions and
     kicks, the last team in control, and the shape of both teams (useful when the ball is missing).
   - *Is the ball dead*: players walk during stoppages (speeds over 1 s, compared with the rest of
     the half), ball out of the pitch, still ball, long gaps that start or end near a line.
3. **Two small logistic regressions** (`possession/model.py`) turn that evidence into "probability
   the ball is dead" and "probability home has it". Their coefficients live in
   `config/possession_v5.json` and were trained **only on public data about other games**.
4. **Decoder** (`possession/decode.py`): an exact Viterbi with a cost per switch, so possessions are
   long and stable instead of flickering, and each boundary lands where the evidence changes.
5. **Write and check** the CSV (`possession/pipeline.py`); before each half's kick-off (Smart
   Tagging) the ball is dead.

## Why not the public 5 minutes

- Only 17 possessions: one missed possession moves spellF1 by about 6 points, and the scorer's own
  noise band there is ~0.41.
- They do not look like the match: the ball is missing on 11.8 % of those frames, against 40.7 % in
  the whole match.
- The organisers show it: the official `smooth125` baseline scores 0.4242 there but 0.2242 on the
  full match.
- The rules (23 Oct) say they are not training data and not a tuning set.

## How we test instead

| data (other games, public) | matches | labels | use |
|---|---|---|---|
| Metrica sample data | 3 | from event data | training + test |
| Sportec Open DFL (Bundesliga) | 7 | from corrected tracking, like the challenge's | training + test |
| SkillCorner open data (A-League) | 20 | from broadcast tracking (a real tracker) | test only |

- Metrica and DFL tracking is degraded to look like ours: ball missing on ~41 % of frames, mostly in
  long gaps (80 % of the missing frames in gaps of 10 s or more; 85 % in our match), noise,
  dropouts and identity swaps.
- **Calibration check**: the official baseline scores 0.246–0.247 in this simulation and 0.2433 on
  the real full match. Same algorithm, same score, so the test is as hard as the match.
- Every comparison is paired (same matches, same noise) with a bootstrap interval.

## Versions

| version | what changed | DFL (spellF1) | SkillCorner | on our match: possessions / dead ball |
|---|---|---|---|---|
| baseline | official `smooth50` | 0.246 | 0.233 | 594 / 32 % |
| v3 | Viterbi decoder, learned dead ball, touch and team-shape evidence | 0.580 | 0.303 | 220 / 61 % (too much dead ball) |
| v4 | dead-ball evidence that does not depend on the tracker's noise; trained on many noise models | 0.589 | 0.341 | 223 / 50 % |
| **v5** | v4 also trained on the 7 DFL matches | **0.598** | **0.351** | 221 / 52 % |

- v4 fixed the main problem of v3 on our match: it was calling 61 % of the match dead ball.
- v5 vs v4: +0.009 on DFL and +0.011 on SkillCorner (both significant); −0.012 on Metrica, whose
  labels count every touch as a possession, unlike the challenge's definition.

## Tried and dropped

SkillCorner as training data, separate weights while the ball is missing, cheaper team switches at
touches, Smart Tagging as team evidence (the reliable tags rarely cover the ball gaps). Details in
`docs/research/v4-robust-dead-ball.md` and `docs/research/v5-dfl.md`.

## Rules we follow

- The match data never leaves the local machine. In the Claude sessions behind this branch only
  aggregate numbers (scores, counts) were shared, never rows of tracking or labels.
- External data, credited in the technical document: Metrica Sports sample data; SkillCorner open
  data (MIT); Sportec Open DFL (CC-BY 4.0, data © Deutsche Fußball Liga — cite Bassek, Rein, Weber
  & Memmert 2025, *Scientific Data* 12, 195, doi:10.1038/s41597-025-04505-y).
- Reproducible: `python scripts/train_model.py <sample-data/data> --version v5 --dfl <dfl folder>`
  regenerates `config/possession_v5.json` byte for byte (checked on Linux; `scripts/fetch_dfl.py`
  downloads DFL).

## Open items

- Team OK on decisions D005–D009 (`docs/DECISIONS.md`).
- Fran's method: put it on a branch and run it through the same external tests.
- Sync the 23 Oct upstream rules (PLAN U1); `AGENTS.md` still says to tune on the public 5 minutes.
- Pin dependencies (Python 3.11, numpy 2.4.6, pandas 3.0.6) and write the technical document.
