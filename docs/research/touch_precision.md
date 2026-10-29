# W3 — touch precision, touch-now evidence and restart clues (2026-10-28)

**Outcome: v5 adopted** (`config/possession.json`), replacing v4. Every number comes from a
command in this repository; nothing was fitted on the competition match and no match content
entered an agent context (guard active, aggregates only).

## 1. Where v4 loses, measured on the external judges

`scripts/short_spell_stats.py` (Metrica sample, 1,022 true possessions) and
`scripts/metrica_eval.py --diagnose`:

| true duration | n | ball seen ≥ 50 % | own touch vote inside | wrong-team vote inside | recall @ IoU 0.5 (v4) |
|---|---|---|---|---|---|
| 0–1 s | 112 | 99 % | 39 % | 38 % | 1 % |
| 1–2 s | 140 | 100 % | 62 % | 36 % | 16 % |
| 2–4 s | 184 | 99 % | 75 % | 35 % | 40 % |
| 4–8 s | 175 | 97 % | 94 % | 45 % | 62 % |
| ≥ 8 s | 411 | 91 % | 98 % | 54 % | 80 % |

Oracle experiments on the same games (true D frames / true team labels injected into the
evidence, everything else as v4): spellF1 0.563 → 0.657 (oracle dead) and → 0.810 (oracle team).
Team ownership, not stoppages, is the larger loss on the external judges; on the public window
the deficit was traced to stoppages (W2), so both matter.

Why short possessions are missed: not detection (77 % of true team changes have a vote of the
new team within −0.5/+1 s, `scripts/touch_precision_stats.py`) but evidence sharpness. The team
logit is capped at ±1.1 nats and the 1.5 s control window and the slow shape clue still favour
the previous owner, so a 1–3 s possession rarely pays the 32 nats of a switch in and out.
Re-weighting alone (w_touch 0.5–2, cap 0.75–0.95, costs 12–24, control 0–1.5) moves every
external score by ≤ 0.01: the plateau is one of information, not of weights.

## 2. The change (v5)

1. **Pass-by filter** (`features.touch_features`, `touch_passby_dv_m_s` 2.0). Touch-vote runs
   during which the ball stays faster than 3 m/s and its velocity (median-filtered positions,
   0.2 s either side) never changes by more than the threshold plus twice the match's own
   median change (a label-free jitter floor) are balls flying past a player, not touches.
   On Metrica they are 11 % of vote runs at 47 % precision; the remaining runs are at 84 %.
2. **Touch-now evidence** (`evidence.w_touch_now` 2.0): while a player is seen playing the ball
   (a surviving vote), 2 nats are added beyond the team cap. A confirmed touch is the label by
   definition; the cap was designed for the proxies.
3. **Shorter minimum touch** (`touch_min_frames` 5 → 3): affordable once pass-bys are filtered.
4. **Restart clues** (`features.restart_features`): `w_out_player` 1.0 — a player standing
   (< 1 m/s) more than 0.5 m beyond a line (thrower, ball retriever; needs no ball; 6–50 times
   likelier on dead frames on every external match, `scripts/restart_clue_stats.py`); and
   `w_slow_far` 1.0 — a seen ball slower than 2 m/s with nobody within 2.5 m for most of a
   second (a ball waiting for a restart; alive false-positive rate 0.0–1.4 % on every judge).
   `w_edge` (slow ball within 2 m of a line) is implemented but off: not supported externally.

## 3. Results

Mean spellF1 over matches; noisy = `external.inject_noise`, seeds 0 and 1.

| Judge | v4 | v5 | change |
|---|---|---|---|
| Metrica clean (3) | 0.563 | **0.605** | +0.042 |
| Metrica noisy (3 × 2) | 0.518 | **0.570** | +0.051 |
| IDSSE clean (7) | 0.508 | **0.576** | +0.068 |
| IDSSE noisy (7 × 2) | 0.397 | **0.435** | +0.038 |
| Metrica / IDSSE macroF1 | 0.770 / 0.808 | 0.779 / 0.832 | +0.009 / +0.024 |
| Public spellF1 [5–95 %] | 0.525 [0.35, 0.70] | 0.533 [0.36, 0.69] | +0.008 |
| Public macroF1 | 0.811 | 0.823 | +0.012 |
| Public possessions predicted / true | 15 / 17 | 16 / 17 | |
| Transplant 10 seeds mean / min | 0.492 / 0.400 | 0.496 / 0.400 | |
| Transition-centred 5 / 10 / 20 s | 0.341 / 0.415 / 0.400 | 0.317 / 0.415 / 0.400 | |
| Full match possessions, mean duration | 314, 12.7 s | 305, 12.7 s | |

Every one of the ten external matches improves, clean and with both noise seeds
(`scripts/py scripts/experiment.py --noise --seeds 0 1 --per-match --public --variant base
--variant v5:evidence.w_touch_now=2.0,features.touch_passby_dv_m_s=2.0,features.touch_min_frames=3,evidence.w_out_player=1.0,evidence.w_slow_far=1.0`
against the v4 config). The public window moves inside its interval; tc5 loses one of its
10 matched possessions, the other masks are unchanged, macroF1 rises on every mask.

Contribution on the external judges (clean / noisy Metrica, IDSSE; seed 0 only):
pass-by filter alone +0.005 / 0, +0.010 / 0; touch-now alone −0.006 / +0.014, +0.013 / +0.008
(sharper evidence needs the cleaner votes); both +0.029 / +0.024, +0.032 / +0.006; plus
out-of-pitch player +0.034 / +0.029, +0.035 / +0.012;
plus slow far ball −0.001 / 0, +0.020 / +0.004 (neutral on Metrica, whose ball is unseen on
78–90 % of dead frames; its target effect is +0.012 public spell, +0.011 macro, +0.013
transplant, and it is adopted as a physically justified rule, not on that evidence);
min_frames 3 +0.011 / +0.015, +0.011 / +0.019.

## 4. Measured and not adopted

| Idea | Why not |
|---|---|
| Minimum dead-stretch duration after decoding (25–100 frames → previous owner) | neutral to −0.005 on both judges; the target has no predicted D run under 1 s |
| Decoder minimum duration (25/50 frames, all states) | −0.001 / −0.006 |
| Sharper or shifted player-speed clue (w_slow 2, ref 1.8, scale 0.35) | −0.005 to −0.02 Metrica |
| dead_bias −0.8 | +0.005 / +0.01 external, but public 0.521 → 0.482 and transplant / tc10 −0.04 / −0.06: a broad target regression |
| Ball at the pitch edge and slow (`w_edge`, the colleague-branch / PR 7 cue) | −0.003 Metrica, +0.012 IDSSE (frozen-ball artefact); off |
| Ball at rest ≥ 1 s, all players in their own half | never fires on the target's tracking (jitter keeps the ball above 0.5 m/s; kick-offs only) |
| w_touch 0.7 / 1.3, decay 10 s, w_control 0.5 / 1.0, w_shape 0.5, radius 1.2 m, cost 20, touch-now 1.5 / 3 | all within ±0.01 of v5 on both judges |

## 5. What the organisers' repository told us (issues and closed PRs, 2026-10-28)

- Issues #10/#14/#16: no labels or parameter choices on the competition match, external open
  data (IDSSE) explicitly allowed with attribution; look-ahead is allowed (#15); intermediate
  outputs may be committed with the script that regenerates them (#14).
- Issue #17 (open): the public ground truth labels a won header without control as a 1.3 s
  possession, so "last touch" is the effective convention; v4/v5 follow it.
- PR 7 (a 3-state HMM, public 0.607 / macro 0.893): its dead evidence is the mean player speed
  with a sharp sigmoid at 1.6 m/s plus an edge-distance term and a sub-1 s run absorption;
  measured here, the sharper speed clue and the edge term do not transfer to the external
  judges and the absorption never fires. Its loose-ball rule (a ball far from everyone is flat
  evidence) is already how our control clue behaves.
- PR 13 (data layer): the ball is missing at the end of every GOALS phase, and 2 % of visible
  ball positions are more than 2 m outside the pitch, only a quarter of them near restarts.

## 6. Verification

- `scripts/py -m unittest discover -s possession -p 'test_*.py'`: 137 tests OK (6 new);
  guard tests 12 OK.
- `scripts/py scripts/verify_submission.py --name ai-enthusiastic`: cold 4.0 s / warm 1.5 s,
  byte-identical (SHA-256 `cfb99407f6abb4f1…`), official validator exit 0, public spellF1
  0.5333 [0.3636, 0.6909], macroF1 0.8228.
- Limitations: the pass-by filter is nearly inactive under the synthetic 0.3 m ball jitter
  (the floor rises with it, by design); the ATD's real jitter is unknown. The out-of-pitch clue
  fires on 4 % of the public window's dead frames (the ATD rarely tracks players beyond the
  lines); the slow-far clue is supported by physics and by the target's alive false-positive
  rate, not by an external gain.
