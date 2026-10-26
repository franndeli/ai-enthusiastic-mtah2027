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
| decode | `decoder.decode(log_evidence, switch_cost, cost_scale=None, min_duration_frames=1) -> (n,) int8` codes H=0 A=1 D=2; sequential Viterbi (the one intentional per-frame loop) |
| structure | `structure.decode_halves(log_evidence, decode_fn) -> (n,) int8`: X=3 outside graded blocks, D before kick-off, halves decoded separately; `decode_fn(block)` or `decode_fn(block, start)` |
| write | `io.write_submission(path, codes)` |

`run.label_codes_from_config(config, match, tags)` wires the stages; `run.restart_cost_scale`
turns the `restart_prior` feature into per-frame cost multipliers when `decoder.restart_discount`
is set.

Parameters live in `config/possession.json`; experiments change values there and record results in
`docs/PLAN.md` (public spellF1/macroF1 with `--bootstrap 1000`, possession count and mean
duration on the full match, stress-test score). Never version experiments by copying files.

Tests: `scripts/py -m unittest discover -s possession -p 'test_*.py'`, synthetic inputs only.
Cached arrays go to `MTAH_WORK_DIR` (set by `scripts/py`), never into the repo.

Rule zero applies to every stage: scripts print aggregates (scores, counts, shapes, timings), never
rows, frames, positions or per-frame labels.
