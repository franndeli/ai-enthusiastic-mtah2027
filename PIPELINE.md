# Our possession pipeline — what we built and how to use it

Short version for the team. Details and every number: `docs/research/` (v3 to v6 notes),
`docs/PLAN.md` and `docs/DECISIONS.md`.

## In one minute

- For each of the 199,319 frames of the tactical video we output `H`, `A` or `D` (and `X` outside
  the graded halves), from the automatic tracking only.
- The current version is **v6**, the default of `run.py`: v5 plus PathCRF, a published model that
  infers possession from the players' movement without the ball. +0.04 spellF1 over v5 on public
  matches neither model saw; on our match it passed every check we fixed beforehand (30 Oct,
  `docs/research/v6-pathcrf.md`).
- We **never tune on the public 5 minutes**. We test on **30 full matches from other teams** that do
  have labels, with their tracking made as bad as ours.
- On the DFL matches v5 scores **0.60 spellF1, against 0.25 for the official baseline**
  (SkillCorner: 0.35 against 0.23); v6 adds about +0.04 on top.

## Test it on your machine (about 10 minutes)

You need Python 3.11 and the match data where `run.py` looks for it: `~/.mtah2027/data/tactical/`
(file names as in `DATA.md`), or pass `--data <folder that contains tactical/>`.

```sh
git fetch origin
git checkout feat/possession-pipeline-v5
git pull

# a fresh environment, so nothing else on your machine changes
py -3.11 -m venv .venv-v6                # Windows (PowerShell)
.\.venv-v6\Scripts\Activate.ps1
python3.11 -m venv .venv-v6              # macOS / Linux
source .venv-v6/bin/activate

pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-v6.txt       # every version pinned

python scripts/check_v6.py               # add --data <folder> if needed
```

`scripts/check_v6.py` checks Python and the pinned versions, runs
`python run.py --out evaluation/results/v6.csv` (a few minutes on CPU; it prints only aggregates)
and compares the file's SHA-256 with the reference in `config/reference_outputs.json`. Expected:

```text
check_v6 PASS
  sha256       c961f5f214912857933721c1661874499825e3b43ad91eade3f4f509d3fae120
  reference    c961f5f2... (Ángel, 2026-10-30, two runs: Windows, Python 3.11.9, ...)
  output       247 possessions, D 51.5 % (reference 247, 51.5 %)
```

PASS means your machine regenerates the submission byte for byte, which the rules require.

**If the check fails**

- *packages missing / different*: install exactly as above, in a fresh venv.
- *match data not found*: pass `--data <folder that contains tactical/>`.
- *FAIL with another SHA-256*: paste the report it prints (OS and versions included) in the team
  chat. Even when possessions and dead-ball share match, a different file has to be fixed before
  submitting, because the organisers must regenerate the same one.
- PowerShell refuses to activate the venv:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force`, then activate again.

## Run it

```sh
python run.py --out evaluation/results/v6.csv                                        # v6 (default)
python run.py --out evaluation/results/v5.csv --model config/possession_v5.json      # v5, no PyTorch
python run.py --out evaluation/results/v4.csv --model config/possession_v4.json      # older versions
python evaluation/run_evaluation.py               # public 5 minutes: a sanity check, not a score
```

- Without the v6 packages `run.py` stops at once and says what to install (or use v5).
- It prints only aggregates (possessions, % dead ball), runs the official `validate_submission.py`
  on the exact file and writes a manifest with the versions to `~/.mtah2027/work/`. No training at
  run time, no randomness: the same SHA-256 on every run.

## Check the claims yourself, on public data (optional)

These use only public matches from other teams, so they also run where the challenge data is not
available.

| what | command | time | expected |
|---|---|---|---|
| unit tests | `python -m unittest discover -s possession -p "test_*.py"` and `... -s scripts ...` | 1 min | 24 + 20 OK |
| v5 vs v6 in the exact submission path, on DFL J03WR9 laid out like the challenge's files | `python scripts/e2e_public.py <dfl folder>` | ~10 min | spellF1 v5 0.5625 → v6 0.6377, four checks `ok` (below) |
| the v6 validation table (5 held-out matches × 3 seeds) | `python scripts/validate_v6.py <dfl folder> <metrica sample-data/data>` | ~1.5 h | +0.044 DFL, +0.037 Metrica (`docs/research/v6-pathcrf.md`) |

Expected output of `e2e_public.py` (Linux, 30 Oct). With the same input file, a different
SHA-256 for v5 or v6 is a reproducibility problem worth reporting:

```text
input  dfl_J03WR9.npz  sha256 9219c61c22db3d44
bundle ...bundle_J03WR9_seed10  (ATD sha256 a9503562106164cf)

| run      | spellF1 | macroF1 | possessions (truth 267) | D share | sha256           |
| v5       | 0.5625  | 0.8478  | 235                     | 43.1 %  | 88c1812ffc96ba9e |
| v6       | 0.6377  | 0.8586  | 263                     | 43.1 %  | ce12d7bc9cdb03d1 |
| v6 again | 0.6377  | 0.8586  | 263                     | 43.1 %  | ce12d7bc9cdb03d1 |

  ok    every output valid
  ok    v6 gives the same SHA-256 twice
  ok    dead ball identical in v5 and v6
  ok    v6 above v5 (spellF1)
```

- DFL data: in a separate venv, `pip install kloppy==3.19.1 numpy==2.4.6 pandas==3.0.6`, then
  `python scripts/fetch_dfl.py <dfl folder>` (all seven matches, ~2.3 GB downloaded, a few MB
  kept per match) or add `--matches J03WR9` for the end-to-end check only.
- Metrica data: `git clone https://github.com/metrica-sports/sample-data` and pass its `data/`.

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
   `config/possession_v6.json` (the same as v5's) and were trained **only on public data about
   other games**.
4. **Decoder** (`possession/decode.py`): an exact Viterbi with a cost per switch, so possessions are
   long and stable instead of flickering, and each boundary lands where the evidence changes.
5. **Second opinion without the ball (v6)** (`possession/retrack.py`, `possession/pathcrf.py`):
   the tracklets are joined into 11 tracks per team, PathCRF's pretrained model estimates which team
   has the ball inside the stretches decoded as alive, that evidence is added to the team model and
   the decoder runs again. Dead ball is unchanged.
6. **Write and check** the CSV (`possession/pipeline.py`); before each half's kick-off (Smart
   Tagging) the ball is dead.

## Why not the public 5 minutes

- Only 17 possessions: one missed possession moves spellF1 by about 6 points, and the scorer's own
  noise band there is ~0.41.
- They do not look like the match: the ball is missing on 11.8 % of those frames, against 40.7 % in
  the whole match.
- The organisers show it: the official `smooth125` baseline scores 0.4242 there but 0.2242 on the
  full match.
- The rules (23 Oct) say they are not training data and not a tuning set.
- We only use them as a sanity check (nothing broken, home and away not swapped): v6 0.7176 / 0.8587
  macroF1, v5 0.5733 / 0.8492.

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
| v5 | v4 also trained on the 7 DFL matches | 0.598 | 0.351 | 221 / 52 % |
| **v6** | v5 + PathCRF ball-free team evidence | **+0.044** on 2 DFL matches held out by PathCRF (Metrica: +0.037) | not applicable (broadcast tracking: −0.02) | 247 / 52 % |

- v4 fixed the main problem of v3 on our match: it was calling 61 % of the match dead ball.
- v6 only helps when nearly all players are tracked with smooth positions; our match's tracking
  report (20.5 objects per frame, jitter 0.0014 m) says it is, unlike SkillCorner's broadcast data.
- v5 vs v4: +0.009 on DFL and +0.011 on SkillCorner (both significant); −0.012 on Metrica, whose
  labels count every touch as a possession, unlike the challenge's definition.

## Tried and dropped

SkillCorner as training data, separate weights while the ball is missing, cheaper team switches at
touches, Smart Tagging as team evidence (the reliable tags rarely cover the ball gaps), formation
contrasts and per-match self-training, a ball detector on the video. Details in
`docs/research/v4-robust-dead-ball.md`, `docs/research/v5-dfl.md` and `docs/PLAN.md` (E8, E9, B3).

## Rules we follow

- The match data never leaves the local machine. In the Claude sessions behind this branch only
  aggregate numbers (scores, counts) were shared, never rows of tracking or labels.
- External data and models, credited in the technical document: Metrica Sports sample data;
  SkillCorner open data (MIT); Sportec Open DFL (CC-BY 4.0, data © Deutsche Fußball Liga — cite
  Bassek, Rein, Weber & Memmert 2025, *Scientific Data* 12, 195, doi:10.1038/s41597-025-04505-y);
  PathCRF code and pretrained weights (MPL-2.0, vendored unmodified in `third_party/pathcrf` — cite
  Kim et al., KDD 2026, doi:10.1145/3770855.3818152).
- Reproducible: `python scripts/train_model.py <sample-data/data> --version v5 --dfl <dfl folder>`
  regenerates `config/possession_v5.json` byte for byte (checked on Linux; `scripts/fetch_dfl.py`
  downloads DFL); `run.py` gives the same file on every run, and `scripts/check_v6.py` checks it
  across machines.

## Head-to-head with the other two pipelines (27 Oct)

All three team pipelines ran unchanged (their own `run.py` and model) on 20 public matches none of
them was trained on (SkillCorner, laid out exactly like the challenge's files), scored with the
official scorer (`scripts/rehearsal.py`, `docs/research/head-to-head-skillcorner.md`):

| pipeline | spellF1 | macroF1 | possessions (truth 293) | D share (truth 51 %) |
|---|---|---|---|---|
| **v5** (this branch) | **0.351** | **0.716** | 259 | 43 % |
| main (Fran) | 0.331 | 0.574 | 309 | 22 % |
| Route A (Javi, tags off) | 0.315 | 0.581 | 416 | 20 % |
| official baseline b50 | 0.233 | 0.655 | 496 | 31 % |

v5 − main +0.021 [+0.011, +0.031] (ahead on 18 of 20 matches); v5 − Route A +0.036 [+0.027,
+0.046] (20 of 20). The difference is mostly dead ball: the other two call ~20 % of frames D.

## Ball-detector pilot (done: no-go, parked)

`scripts/ball_detector_pilot.py` runs a COCO-pretrained YOLO on a slice of the tactical video and
prints aggregates only. On two 3,000-frame slices its top detection was within 2 m of the ATD ball
on only 53–59 % of frames, so its gap fills cannot be trusted (`docs/PLAN.md`, B3).

## Open items

- Fran runs `python scripts/check_v6.py` on his machine and gets PASS (`docs/PLAN.md`, R1) before
  the file is submitted.
- Team OK on decisions D005–D010 (`docs/DECISIONS.md`).
- The other pipelines: check their full-match D share (aggregate); compliance of public-window
  calibration with the 23 Oct rules (head-to-head note).
- Sync the 23 Oct upstream rules (PLAN U1); `AGENTS.md` still says to tune on the public 5 minutes.
- Write the technical document (dependencies are pinned in `requirements-v6.txt`).
