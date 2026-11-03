---
id: architecture
type: architecture-design
status: draft
title: Pipeline contract and module boundaries
parent: goal
references:
  - data-policy
  - coding
---

# Pipeline contract

One command regenerates the submission:

    scripts/py run.py --out evaluation/results/<name>.csv [--config config/possession.json]

Stages, each a pure function over numpy arrays (`possession/`). Match I/O lives in `io.py`;
external-data I/O in `external.py` (IDSSE), `metrica.py` (Metrica sample data) and `learned.py`
(model files); nothing else touches files.

| Stage | Signature |
|---|---|
| load | `io.load_match(source="tactical") -> Match`: arrays per video frame, `players_xy (n,46,2)`, `players_speed (n,46)`, `ball_xy (n,2)`, `ball_speed (n,)`, NaN = missing; `team_of_slot (46,)` HOME=0/AWAY=1; `halves`; `pitch_m`. `io.load_smart_tagging() -> list` of tagged intervals. |
| features | `features.compute(match, tags=None, params=None) -> dict[str, (n,) float32]`, NaN = not computable on that frame; `params` = `config["features"]` overrides of the touch / ball-motion rule constants |
| evidence | `evidence.scores(features, params) -> (n,3) float32` log-evidence for H, A, D; never NaN. `learned.scores` has the same signature. |
| decode | `decoder.decode(log_evidence, switch_cost, initial=None, *, cost_scale=None, min_duration_frames=1) -> (n,) int8` codes H=0 A=1 D=2; sequential Viterbi (the one intentional per-frame loop) |
| structure | `structure.decode_halves(log_evidence, decode_fn) -> (n,) int8`: X=3 outside graded blocks, D before kick-off, halves decoded separately; `decode_fn(block)` or `decode_fn(block, start)` |
| write | `io.write_submission(path, codes)` |

`run.label_codes_from_config(config, match, tags, models)` wires the stages; `run.restart_cost_scale`
turns the `restart_prior` feature into per-frame cost multipliers when `decoder.restart_discount`
is set.

v6 (a `"v6"` section in the config, D015) changes the middle of the path, in `run.v6_codes`:

| Stage | Signature |
|---|---|
| logits | `evidence.team_logit(features, p)`, `evidence.dead_logit(features, p)`; `evidence.combine(z_team, z_dead, p) -> (n,3)` |
| dead residual | `dead_residual.residual(model, match, features, z_dead_rule) -> (n,)` clipped to +-1, added to the dead logit |
| PathCRF | `pathcrf.home_probability(match, alive) -> (n,)` inside v5's alive stretches; `pathcrf.logodds(p)` x `v6.w_pathcrf` added to the team logit (torch, CPU) |
| verifier | `verifier.decode_block(model, block_evidence, features, start, verifier.viterbi_decode(cost)) -> codes`, the `decode_fn` of `structure.decode_halves` |

`run.load_models(config)` reads `models/*.pkl` through `io.load_model`, which checks the SHA-256
recorded in the config. Models are trained only by `scripts/train_v6.py` on external data.

Parameters live in `config/possession.json`; experiments change values there and record results in
`docs/PLAN.md` (public spellF1/macroF1 with `--bootstrap 1000`, possession count and mean
duration on the full match, stress-test score). Never version experiments by copying files.

Tests: `scripts/py -m unittest discover -s possession -p 'test_*.py'`, synthetic inputs only.
Cached arrays go to `MTAH_WORK_DIR` (set by `scripts/py`), never into the repo.

## Modules

| Spec id | Owns | May depend on |
|---|---|---|
| `possession-pipeline` | the production path above (`io`, `features`, `evidence`, `decoder`, `structure`, `dead_residual`, `pathcrf` + `retrack`, `verifier`, `run.py`) | upstream `games.py`, vendored `third_party/pathcrf` |
| `external-judges` | public external matches as `Match` + labels, noise, and the scripts that score on them | `possession-pipeline`, upstream `evaluation/` scorer |
| `evidence-experiments` | alternative evidence models kept outside production (`learned`, `robust_dead`, `broadcast`) | `possession-pipeline`, `external-judges` |
| `data-guard` | optional agent tool-call guard | nothing in the package |

Nothing in `possession-pipeline` imports a judge or an experiment; `run.py` wires only the
production stages. Upstream files (`games.py`, `evaluation/`, `starter/`) are reused, not specced.

Rule zero applies to every stage: scripts print aggregates (scores, counts, shapes, timings), never
rows, frames, positions or per-frame labels.
