---
id: evidence-experiments
type: module-design
status: draft
title: Evidence experiments (unwired)
parent: architecture
depends-on:
  - possession-pipeline
  - external-judges
references:
  - decisions
---

# Evidence experiments

## Responsibility

Hold alternative evidence models that were tested and not adopted, so their results can be
reproduced:

- `learned.py` is a gradient-boosted evidence model trained on noisy IDSSE. It has the same contract
  as `evidence.scores` and is direction-agnostic by construction. Not adopted (D006, D008).
- `robust_dead.py` is the BR2 player-only activity head (`task-br2-tracker-robust-dead`). Not
  adopted (D013).
- `broadcast.py` is the formation-only broadcast-to-tactical clock alignment (W7a). It was dropped
  because it recovered almost none of the ball gaps (`technical-doc` §2).

Their runners are `scripts/train_evidence.py`, `learned_eval.py`, `robust_dead_eval.py` and
`broadcast_align.py`.

## Boundary

- `run.py` and `possession-pipeline` never import these modules. They may import the pipeline
  stages and `external-judges`.
- Training uses external data only. A model file may be committed only with its hash and its
  training command (D005).
- Promoting an experiment into production means passing the gate in `goal` and recording a new
  decision. Retuning a rejected recipe on the public window is forbidden (D012, D013).

## Limitations

- This code is kept in the package although production does not use it, which is in tension with
  `coding`'s "delete dead code". It is retained because D006 and D013 decide to keep it as
  documented experiments. For `broadcast.py`, the reason it is still kept is *unconfirmed*.
