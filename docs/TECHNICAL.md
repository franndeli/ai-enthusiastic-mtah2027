---
id: technical-doc
type: architecture-design
status: draft
title: Technical document (submission deliverable)
parent: goal
references:
  - architecture
  - decisions
---

# ai-enthusiastic: MTAH 2027 Phase 1

Repository: https://github.com/franndeli/ai-enthusiastic-mtah2027

## Method

Our pipeline labels each frame as home possession (`H`), away possession (`A`), dead ball (`D`)
or ungraded (`X`). It uses the tactical tracking, with rules for visible-ball situations and
learned evidence for gaps. The ball is missing on about 41% of graded frames, so nearest-player
assignment alone is insufficient.

The rules combine sustained control, the last team to touch the ball, team shape, player speeds
and ball exits. A pass-by filter rejects apparent touches when the ball flies past a player
without changing velocity. Loose balls retain the previous team's evidence, which fades during
long gaps. Ball absence alone does not mean dead ball. Stoppages and restarts are inferred from
player movement, ball motion and proximity to the pitch boundary.

Three learned components supplement these rules:

- **PathCRF** infers the possessor from player movement when ball evidence is weak. Fragmented
  tracking is reconstructed into 11 tracks per team before inference.
- A gradient-boosted model corrects dead-ball evidence using player-only context.
- A second gradient-boosted model scores alternative possession segments proposed by the decoder.

A Viterbi decoder selects a sequence for each half, penalising unnecessary switches. Possessions
never cross halftime. Inside the graded halves the system answers H, A or D; outside them it
writes X. It does not assign ownership during dead ball.

We do not interpolate the ball across long gaps or treat tracking slots as stable player
identities. The final parameters and model hashes are in `config/possession.json`. All production
parameters were selected on external data; no output was hand-corrected.

## Data and models

The final pipeline uses tactical ATD. Smart Tagging is disabled because we had no external tagged
data to tune its weight. Broadcast tracking was dropped after alignment recovered only 0.5% of
tactical ball gaps. Neither video nor homography is used in the final method; a video ball-detection
pilot did not meet its agreement check.

Training labels come from the providers, not from our team's annotations:

- **IDSSE / Sportec Open DFL:** seven Bundesliga matches, Bassek et al. (2025), CC-BY 4.0.
  Provider `BallStatus` and `BallPossession` fields supply the labels. Used to fit team-shape
  weights, train the two correction models and evaluate the method. Fetch: `scripts/fetch_idsse.py`.
- **Metrica Sports sample data:** three anonymised matches from
  https://github.com/metrica-sports/sample-data. Labels are derived from synchronised provider
  events. Used to train the correction models and evaluate the method.
- **PathCRF:** Kim et al., KDD 2026, https://github.com/hyunsungkim-ds/pathcrf, MPL-2.0.
  We use the authors' checkpoint trained on five Sportec matches. Code, checkpoint and licence
  are included under `third_party/pathcrf/`.

Twenty SkillCorner A-League matches from https://github.com/SkillCorner/opendata (MIT) provide
an additional evaluation set, with labels derived from provider events and phases. They were
not used to train or tune the final model.

No competition-match labels were used to fit the final pipeline. The public five-minute window
was used to check it. Match data and frame-level outputs are excluded from the repository.

## Results and limitations

On the public five minutes, the final model scores **spellF1 0.6235** and **macroF1 0.8300**.
The spellF1 bootstrap interval is 0.4588–0.7765. This window contains only 17 possessions and
cannot establish full-match performance; the private score is unknown.

On ten held-out SkillCorner matches, spellF1 improved from 0.424 for the previous rule-based
configuration to 0.528, with gains on all ten matches. These are provider-derived proxy labels,
not corrected possession ground truth. Earlier Metrica and IDSSE comparisons excluded evaluated
matches from learned-component fitting, but reused them for rule tuning.

The main errors are missed possessions under two seconds, uncertain hand-overs during long ball
gaps, and stoppages where the tracker still reports a slow ball near players. Deflections can
also be mistaken for controlled touches. With more time, we would focus on short-possession
recall and stoppage detection, tested across providers rather than tuned to the public window.

## Reproduce

Unpack the supplied match bundle outside the repository and set `MTAH_DATA_DIR` to its directory.
Set `MTAH_WORK_DIR` to a separate directory for caches, then run:

```bash
scripts/setup-env.sh
scripts/py run.py --out evaluation/results/ai-enthusiastic.csv
scripts/py evaluation/validate_submission.py evaluation/results/ai-enthusiastic.csv
```

Dependencies are pinned; trained models are included. Retraining is not required to reproduce
the submission. Training procedures are in `scripts/train_v6.py` and `scripts/fit_shape_weights.py`.

`scripts/py scripts/verify_submission.py --name ai-enthusiastic` regenerates the file with and
without caches, compares the bytes and runs the official validator. Both runs passed, with
SHA-256 prefix `263bfddea0d0eeb7`. The same output was verified on macOS arm64 and Linux x86_64.
