# ATD vs DFL domain gap and ATD-like ball dropout (2026-10-26)

Command: `scripts/py -m possession.domain_gap` (aggregates only).

| statistic (play frames) | ATD BAR-ATH | DFL mean of 7 |
| --- | --- | --- |
| ball missing share | 0.407 | 0.000 |
| ball gap mean / median / p90 (s) | 17.2 / 10.7 / 33.8 | — |
| players seen | 20.49 | 21.81 |
| player speed p50 / p90 (m/s) | 1.36 / 3.81 | 1.40 / 3.81 |
| ball-loss hazard per frame, 0 / 1 / 2 / 3+ players within 5 m | .0014 / .0015 / .0015 / .0024 | — |

Public window (public GT x ATD, aggregates): P(ball missing | D) 0.336, P(missing | alive) 0.065;
P(D | missing) 0.558 vs P(D | seen) 0.147.

## Reading

- Position noise and player coverage are close to DFL; the crowding effect on ball loss is mild.
- The real gap is the ball: ATD gaps are long (median 10.7 s) and concentrated in dead ball. The
  first dropout (geometric blocks of 1.5 s, uniform in time) erased that signal.
- `train.drop_ball` now draws lognormal gap lengths (median 10.7 s, sigma 0.97 fits mean 17.2 s
  and p90 ~34 s) and starts gaps 5x more often in dead ball (0.336 / 0.065), at 40.7% overall.

## Cross-evaluation (DFL LOMO, every model scored under the ATD-like gaps)

Mean spellF1 (macroF1) over 7 held-out matches; nearest-player smooth50: 0.2303 (0.526).

| training corruption | vit 0.2 | vit 0.5 | vit 1.0 | vit 2.0 |
| --- | --- | --- | --- | --- |
| old: short uniform gaps | 0.2916 (0.765) | 0.3116 (0.763) | 0.3181 (0.762) | 0.3132 (0.760) |
| **ATD-like** | 0.3041 (0.774) | **0.3244 (0.773)** | 0.3335 (0.772) | 0.3283 (0.770) |
| ATD-like + gap timers capped at 30 s | 0.3045 (0.772) | 0.3231 (0.771) | 0.3333 (0.769) | 0.3274 (0.768) |

- ATD-like training wins by +0.012 to +0.015 at every scale (paired folds; below the 0.04 band).
- Longer gap timers add nothing; `features.MAX_GAP_S` stays 10 s.
- Under realistic gaps the honest external estimate is ~0.32-0.33, not the 0.40 of the first table
  in `lomo_dfl.md`, which used short gaps.

## Emission scale on the match (aggregate sanity, D007)

| model | full-match possessions | public spells (GT 17) | public spellF1 [5-95%] | public macroF1 |
| --- | --- | --- | --- | --- |
| first candidate (old dropout, 0.5) | 375 | 19 | 0.422 [0.267, 0.578] | 0.785 |
| ATD-like, 1.0 | 472 | 25 | 0.419 [0.285, 0.552] | 0.752 |
| **ATD-like, 0.5 (current)** | 387 | 21 | 0.400 [0.253, 0.547] | 0.744 |

The public window ~extrapolates to ~360 possessions (17 per 7,500 graded frames; Poisson ±25%).
DFL labels change possession more often (370-511 per match), so the DFL optimum (1.0)
over-fragments here; 0.5 is within LOMO noise of it and matches the count check. All public
differences above sit inside the bootstrap intervals: the public window does not rank them.
