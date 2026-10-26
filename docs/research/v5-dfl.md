# v5: training with the Sportec Open DFL matches

Status: default of `run.py` since 2026-10-26 (`config/possession_v5.json`), after the local check
on the match passed (below). v4 stays available with `--model config/possession_v4.json`. All
numbers are aggregates on public data about other games, except the local check, which is
aggregates on the match run on our own machine.

## Data

Sportec Open DFL / IDSSE: seven 2022/23 Bundesliga and 2. Bundesliga matches with TRACAB positions
at 25 Hz and, on every frame, `BallStatus` (alive / dead) and `BallPossession` (home / away).
Licence CC-BY 4.0; data © Deutsche Fußball Liga (DFL). Cite: Bassek, M., Rein, R., Weber, H. &
Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer.
*Scientific Data* 12, 195. https://doi.org/10.1038/s41597-025-04505-y. Raw files:
https://doi.org/10.6084/m9.figshare.28196177, mirrored at huggingface.co/datasets/pysport/idsse-data.

`scripts/fetch_dfl.py` (kloppy 3.19.1, in its own venv) downloads each match and saves it as
`dfl_<match>.npz` (positions in cm, per-frame H/A/D), ~12 MB each; `possession/dfl.py` reads them.

Per match: 137–147 k frames, 22 players per frame, ball always tracked, D 41–52 % of frames.

## Labels: "controlled possession"

DFL's `BallPossession` changes on every touch: 370–511 possessions per match, 94–191 of them under
2 s. CHALLENGE.md counts a change only when the new team gains *controlled* possession ("a player
keeps the ball for a few seconds"; a touch, a deflection or a won aerial duel without control does
not count). `dfl.controlled` keeps the ball with the team that had it when the other team's run is
shorter than 2 s: 267–315 possessions per match (mean 8.7–12.8 s). For comparison, the public
window has 17 possessions of mean 14.2 s (DATA.md) and v4 predicts 223 of mean 14.1 s on the match.
The raw labels are kept for reference.

## Results (`scripts/validate_dfl.py`)

DFL matches degraded like the ATD, each held out in turn (7 × 3 copies), controlled labels:

| model | spellF1 | macroF1 | possessions (truth 288) | D error |
|---|---|---|---|---|
| b50 (official baseline) | 0.246 | 0.614 | 515 | −13.0 pp |
| v3 | 0.580 | 0.850 | 264 | −0.7 pp |
| v4 | 0.589 | 0.853 | 257 | −0.6 pp |
| **v5** (Metrica + other 6 DFL, cost_team 12) | **0.598** | 0.856 | 239 | +0.3 pp |

- v4 − v3 = +0.008 [+0.003, +0.013]: a third independent benchmark (another league, optical
  tracking) agrees with Metrica and SkillCorner. With smooth player positions (scratch run) v4 − v3
  is +0.063 [+0.055, +0.070], v3 over-calling D by 11 pp.
- v5 − v4 = **+0.009 [+0.005, +0.013]**.
- cost_team for v5: 8 → 0.5967, 12 → 0.5978, 16 → 0.5963 (16 − 12 = −0.002 [−0.003, +0.000]).
  12 is shipped.
- The official baseline scores 0.246 here against 0.2433 published for the match, and v4's D share
  error is under 1 pp: the degraded DFL copies behave like the match.

SkillCorner (20 matches, real broadcast tracking), shipped configs: v4 0.3405, v5 0.3510,
**+0.011 [+0.008, +0.015]**; D error −13.5 → −8.5 pp.

Metrica games held out: v4 0.562, v5 0.550, **−0.012 [−0.019, −0.005]**. These labels are derived
from events and count every touch by the other team as a possession (381 per game, mean 8.8 s),
so they reward more switching than the challenge's definition; v5's higher switching cost costs
there. Retraining on Metrica labels closer to "controlled possession" did not help v4 either
(v4 research note).

Also tried (scratch runs, two training seeds): DFL only, no Metrica: +0.007 on DFL, +0.014 on
SkillCorner, −0.016 on Metrica; costs chosen on all training copies instead of the DFL ones:
+0.008 / +0.010 / −0.007. Against the raw, uncontrolled DFL labels the DFL gain disappears
(+0.000 for the latter variant).

## Reading

v5 is ahead on the two benchmarks whose labels and tracking are closest to the challenge (DFL:
possession from corrected tracking, like the GT; SkillCorner: a real tracker) and behind on the one
that is furthest (event-derived, very fragmented labels). Expected effect on the match: small and
positive (~+0.01), not certain. v5 predicts fewer possessions than v4 (DFL 239 vs 257, SkillCorner
260 vs 290) and calls slightly more D (+0.9 pp on DFL, +5 pp on SkillCorner).

## Check on the match before switching the default (local, aggregates only)

Result (2026-10-26, Ángel's machine): valid; same SHA-256 on two runs; **221 possessions** (v4:
223), mean 13.9 s, median 9.5 s; **D 51.5 %** (v4: 50.4 %), i.e. the DFL-like shift (+0.9 pp), not
the SkillCorner-like one (+5 pp). Gap report close to v4 (D with the ball seen 24.4 % vs 23.0 %; in
gaps of 1–10 s 71.2 % vs 68.5 %; in gaps ≥ 10 s 94.7 % vs 94.2 %). Public 5 minutes (sanity only):
spellF1 0.5733 (v4 0.5733), macroF1 0.8492 (v4 0.8669), 13 possessions (truth 17). Passed: v5
becomes the default.

Criteria set beforehand:

- Valid, same SHA-256 twice.
- Expected from the benchmarks: ~200–210 possessions (v4: 223) and D ~51–55 % (v4: 50.4 %).
  Much fewer possessions or D well above 55 % would point to a tracker mismatch: keep v4.
- `gap_report.py` for v4 and v5; public 5 minutes as a sanity check only.

## Reproduce

```sh
python scripts/fetch_dfl.py ~/.mtah2027/external/dfl                  # venv with kloppy 3.19.1
python scripts/train_model.py <sample-data/data> --version v5 --dfl ~/.mtah2027/external/dfl
python scripts/validate_dfl.py ~/.mtah2027/external/dfl <sample-data/data> \
    --skillcorner <opendata>/data/matches
```
