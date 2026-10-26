# W2 — last-touch ownership and a provider-matched external judge (2026-10-26)

**Outcome: v4 adopted** (`config/possession.json`), replacing v3c. Every number is from a command
in this repository; nothing was fitted on the competition match, and no match content entered
an agent context (aggregates only).

## 1. What was wrong with v3c, read off the public ground truth

The public 5 minutes (public file, `evaluation/gt/`) contain possessions of 0.76 s, 1.32 s and
four of 2.8–3.6 s: six of 17 are under 4 s. A single touch by the other team is a possession;
the owner then stays through the flight and loose ball until the next touch by the opponent.
v3c modelled *sustained control* (1.5 s centred window, ball ≤ 8 m/s) with symmetric memory that
places a hand-over at the midpoint of a gap. It predicted 11 of 17 public possessions and 264 on
the full match, and on `scripts/diagnose_public.py` two thirds of its team transitions were more
than 2 s off. The same pattern showed on every external match (recall@0.5: 0.0 for spells under
2 s, 0.1–0.2 for 2–4 s, 0.5 for 4–8 s).

## 2. A better external judge: Metrica Sports sample data

All previous waves (V1–V4) used IDSSE/Sportec, whose labels are finer than ours (D share 41–52%,
mean spell 7.3 s), so every candidate that learned its granularity over-fragmented the target.
`scripts/fetch_metrica.py` + `possession/metrica.py` add the three public Metrica sample matches
(github.com/metrica-sports/sample-data): **same provider as the competition data**, 25 fps,
normalised 0–1 coordinates on 105×68 m, event data synchronised to the tracking. Labels are
derived from the events with the last-touch semantics above (owner = team of the latest set
piece / pass / carry / recovery / shot / ball-lost event; dead from ball-out / foul / goal /
offside to the next set piece); synthetic tests in `possession/test_metrica.py`. They are
event-derived, not corrected-tracking truth, and the three games are anonymised 2019–21 matches
with player identities; the ball is nevertheless missing on 32–41% of frames, like ours.

| Judge | spells / match | mean | median | < 4 s | dead share | ball missing |
|---|---|---|---|---|---|---|
| Public window (first 5 min) | 17 | 14.2 s | 7.6 s | 35% | 19.6% | 12% |
| Metrica sample ×3 | 327–368 | 10.4–11.7 s | 5.1–5.5 s | 40–44% | 29–41% | 32–41% |
| IDSSE ×7 | 370–511 | 5.5–9.2 s | — | — | 41–52% | 0% (frozen ball when dead) |

`scripts/metrica_eval.py` (clean / noisy, `--diagnose`, `--semantics`) and
`scripts/experiment.py` (config variants across judges in one process) are the tooling.

## 3. The change (v4)

1. **Last-touch ownership** (`features.touch_features`, `evidence.w_touch`): a touch vote is the
   nearest team when a visible ball is within 1.5 m of one of its players, the other team at
   least 0.5 m farther, sustained for 5 frames. `owner_sign` is the last vote (causal), fading
   with a 20 s constant and, additionally, with a 5 s constant of the time the ball has been
   unseen so that the ball-free shape evidence takes over in long gaps. The old symmetric gap
   memory (`w_gap`) is off; control (`w_control`) and shape (`w_shape`) are unchanged.
2. **Ball-motion alive clue** (`features.ball_motion_features`, `evidence.w_moving` 1.5): share
   of a 1 s window with the ball seen travelling faster than 5 m/s lowers the dead log-odds.
   Players slow down only after the whistle and speed up only after the restart kick; the ball
   does both first. Validated on Metrica, whose ball is real during stoppages (IDSSE's is frozen).
3. **Switch costs** 16 / 8 / 8 (team / to-dead / from-dead), from 6 / 6 / 6: the touch evidence
   is sharper, so a higher cost removes flicker without losing short spells. (A team cost above
   to-dead + from-dead is routed through D, so 16 is the effective maximum.)

## 4. Results

Mean spellF1 over matches; noisy = `external.inject_noise` (ball jitter 0.3 m, slot
fragmentation, extra ball dropout: 41% on IDSSE, 15% on Metrica on top of its own 32–41%).

| Judge | v3c | v4 | change |
|---|---|---|---|
| Metrica clean (3) | 0.461 | **0.563** | +0.10 |
| Metrica noisy (3 × seeds 0,1) | 0.439 | **0.528** | +0.09 |
| IDSSE clean (7) | 0.414 | **0.508** | +0.09 |
| IDSSE noisy (7 × seeds 0,1) | 0.344 | **0.401** | +0.06 |
| Metrica macroF1 / IDSSE macroF1 (clean) | 0.752 / 0.822 | 0.772 / 0.853 | +0.02 / +0.03 |
| Public spellF1 [5–95%] | 0.600 [0.39, 0.78] | 0.525 [0.35, 0.70] | −0.075 |
| Public macroF1 | 0.827 | 0.811 | −0.016 |
| Public possessions predicted / true | 11 / 17 | 15 / 17 | |
| Public spellF1 with oracle dead frames (diagnostic) | 0.600 | 0.600 | |
| Transplant 10 seeds mean / min | 0.559 / 0.430 | 0.492 / 0.400 | −0.07 |
| Transition-centred 5 / 10 / 20 s | 0.400 / 0.415 / 0.477 | 0.341 / 0.415 / 0.400 | |
| Full match possessions, mean duration | 264, 14.6 s | 314, 12.7 s | |

Every one of the ten external matches improves, clean and noisy, both seeds (per-match table:
`scripts/py scripts/experiment.py --noise --seeds 0 1 --per-match --variant v4`). The
public-window decrease is inside its own interval and is entirely dead-ball handling: giving
v4 the true D frames returns exactly 0.600. On the public D stretches the automatic tracking
keeps seeing a slow ball with players of both teams around it (ball seen on 66% of public D
frames versus 10–22% on Metrica's dead frames; `scripts/dead_stats.py`,
`scripts/dead_stretches.py`), so touch votes inside missed stoppages become spurious short
spells (4 of the 6 spurious public spells). v3c missed the same stoppages but merged them into
a neighbour instead of splitting them. The dead-ball side is therefore the residual risk for
both models, and the next piece of work.

Parameter choices were made on the external judges (coarse grids; `w_touch` 0.5/1/1.5/2, decay
10/20/40, radius 1/1.5/2, margin 0.25/0.5/1, min frames 1/3/5, moving threshold 3–6 m/s, costs
4–20). Where the external judges were flat (costs 10–16, min frames 3–5) the more conservative
end was chosen because the public window shows a lower possession rate (4.2 per alive minute)
than the Metrica games (5.7–6.4); that is the only use of the public window in the selection.

## 5. Rejected in this wave (all measured, none adopted)

| Idea | Why rejected |
|---|---|
| Symmetric gap memory kept alongside touch (`w_gap` 0.3–1.0) | −0.01 to −0.05 on both external judges with v4 costs |
| Next-touch look-ahead inside ball gaps (`w_next_touch`) | −0.003 to −0.008 everywhere |
| Touch radius 1.0 m | +0.02 Metrica, −0.005 IDSSE; kept 1.5 m for jitter robustness |
| Ball-at-rest dead clue (`w_rest`) | +0.01 IDSSE only (frozen-ball artefact), neutral Metrica, no public change |
| Ball seen but slow over 5 s as dead clue (`w_ball_slow`, `w_quiet` with slow players) | +0.03–0.06 public, +0.01 IDSSE, −0.02 Metrica (false D in slow build-ups); conflict, not adopted |
| Touch votes requiring recent ball travel (stoppage gate) | no effect |
| Team-switch cost scaled by p(dead) | no effect |
| Long-gap-with-slow-players dead clue (`w_gap_dead` 0.5–1) | +0.003 / +0.002, noise |
| Shape evidence off / doubled | −0.035 / −0.02 |

## 6. Verification

- `scripts/py -m unittest discover -s possession -p 'test_*.py'`: 108 tests OK (14 new:
  touch, ball motion, evidence terms, Metrica labels). Guard tests 12 OK.
- `scripts/py run.py --out evaluation/results/ai-enthusiastic.csv`: 1.2 s warm, 3.7 s cold;
  official validator OK; cold and warm generations byte-identical (SHA-256 `1f69e8429b62f25d…`).
- Limitations: Metrica labels are event-derived; its three games have player identities; the
  ATD-specific behaviour during stoppages (ball tracked, players near it) is not represented in
  any external judge, so the public window is the only measurement of it (4 stretches).
