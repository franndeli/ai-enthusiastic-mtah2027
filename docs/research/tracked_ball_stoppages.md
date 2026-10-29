# E5 — stoppages with a tracked ball (2026-10-28)

**Outcome: no production change.** The regime is measured, two candidate clues are implemented
and left off (`w_quiet`, `w_edge`), and the reason it cannot be fixed without tuning on the
match is now concrete. Aggregates only; guard active throughout.

## 1. What the target's stoppages look like

`scripts/tracked_dead_stats.py`, `scripts/dead_clue_stats.py` (public window, 4 D stretches,
1,469 frames) against the three Metrica sample games:

| ball-seen frames | target dead | target alive | Metrica G1 dead | Metrica G2/G3 dead |
|---|---|---|---|---|
| ball seen | 66 % | 94 % | 22 % | 10–15 % |
| nearest player, median | 2.9 m | 1.7 m | 1.4 m | 1.3–1.4 m |
| ball speed (4-frame), median | 2.7 m/s | 5.8 m/s | 1.4 m/s | 6.6–6.9 m/s |
| ball travel over 2 s, median | 5.6 m | 12.0 m | 2.6 m | 11.6–11.8 m |
| players' mean speed, median | 1.2 m/s | 2.3 m/s | 1.4 m/s | 2.0–2.2 m/s |

On the target the ball keeps being reported during a stoppage, **moving at walking pace about
3 m from the nearest player** (carried, or a phantom detection), while the players walk. Only
11 % of its dead frames have the ball under 2 m of travel in 2 s, so "ball at rest" clues never
fire there (and its instantaneous speed carries 2–3 m/s of jitter: `scripts/speed_domain_stats.py`
shows the same floor on player speeds, which never fall below 0.5 m/s the way external tracking
does). Metrica Game 1's seen-ball dead frames resemble the target; Games 2 and 3's are
indistinguishable from alive play (event-derived label slop at the stoppage edges), so the
external judge cannot reward a clue for this regime and may punish it.

## 2. Clues measured (1–2 s window shares; LR = P(clue | dead) / P(clue | alive))

| clue | target dead / alive | Metrica G1 | G2 | G3 |
|---|---|---|---|---|
| players beyond the lines, standing (`w_out_player`, adopted in v5) | 2.3 % / 0.0 % | 13 % / 2.1 % | 10 % / 1.5 % | 10 % / 1.3 % |
| slow ball nobody within 2.5 m (`w_slow_far`, adopted in v5) | 9.8 % / 1.4 % | 5.0 % / 0.0 % | 0.0 % / 0.2 % | 0.3 % / 1.0 % |
| ball travel < 8 m in 2 s, nobody within 2 m, players < 1.8 m/s (`w_quiet`) | 9.3 % / 0.0 % | 4.9 % / 0.0 % | 0.4 % / 0.1 % | 0.0 % / 0.6 % |
| ball travel < 8 m in 2 s and players < 1.6 m/s | 28 % / 3.2 % | 11 % / 6.9 % | 0.8 % / 6.7 % | 0.8 % / 4.7 % |
| slow ball within 2 m of a line (`w_edge`) | 4.8 % / 2.1 % | 2.7 % / 1.5 % | 0.0 % / 1.2 % | 0.4 % / 1.5 % |
| ball at rest ≥ 1 s; all players in their own half | 0 % / 0 % | — | — | — |

The conjunction clues are nearly free of alive false positives everywhere, but cover at most
a tenth of the target's dead frames; the broad "slow ball and walking players" clue that would
cover a third of them is wrong on 3–7 % of alive frames on every judge (slow build-ups), which
is the W2 rejection again.

## 3. Pipeline experiments (seed 0; v5 reference Metrica 0.589 / IDSSE 0.507 / public 0.533)

| variant | Metrica | IDSSE | public spell / macro |
|---|---|---|---|
| `w_quiet` 1 / 2 / 3 | −0.001 / −0.005 / −0.007 | +0.009 / +0.009 / +0.007 | 0 / 0 / −0.016; macro +0.008 / +0.010 / +0.009 |
| jitter-robust `ball_moving` (travel over ±0.5 s instead of 4-frame speed) | −0.008 | 0.000 | 0 / −0.004 |
| `dead_bias` −0.8 (W3) | +0.005 | +0.010 | −0.04, transplant −0.04, tc10 −0.06 |

Per stretch (`scripts/dead_stretches.py`), the two long tracked-ball stoppages sit at a mean
dead log-odds of −0.05 and −0.8: the first needs a nudge of a few tenths of a nat on a third of
its frames, the second a real new observation. Nothing external can license either.

## 4. What would unblock it

- A corrected-tracking judge from the same provider with the ball reported during stoppages
  (the competition's own data format) — none is public.
- The organisers' corrected ground truth for more than the 5 public minutes — not available
  before the deadline.
- A ball-detector on the tactical video trained on external footage (allowed, issue #2) that
  reports *no ball* when the ball is being carried off the pitch — a separate project.
