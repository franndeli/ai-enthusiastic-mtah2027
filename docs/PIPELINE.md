# Pipeline contract

One command regenerates the submission:

    scripts/py run.py --out evaluation/results/<name>.csv [--config config/possession.json]

Stages, each a pure function over numpy arrays (`possession/`), I/O only in `io.py`:

| Stage | Module | Signature | Owner |
|---|---|---|---|
| load | `io.load_match(source) -> Match` | arrays per video frame: `players_xy (n,46,2)`, `players_speed (n,46)`, `ball_xy (n,2)`, `ball_speed (n,)`, NaN = missing; `team_of_slot (46,)` HOME=0/AWAY=1; `halves`; `pitch_m` | coordinator |
| features | `features.compute(match) -> dict[str, (n,) float32]` | NaN = not computable on that frame | T1 |
| evidence | `evidence.scores(features, params) -> (n,3) float32 log-evidence` for H, A, D; never NaN | T1 (rules), T5 (learned, same signature) |
| decode | `decoder.decode(log_evidence, switch_cost, ...) -> (n,) int8` codes H=0 A=1 D=2 | T2 |
| structure | `structure.decode_halves(log_evidence, decode_fn) -> (n,) int8` with X=3 outside graded blocks, D before kick-off, halves decoded separately | T2 |
| write | `io.write_submission(path, codes)` | coordinator |

Parameters live in `config/possession.json`; experiments change values there and record results in
`docs/PLAN.md` (public spellF1/macroF1 with `--bootstrap 1000`, possession count and mean
duration on the full match, stress-test score). Never version experiments by copying files.

Tests: `scripts/py -m unittest discover -s possession -p 'test_*.py'`, synthetic inputs only.
Cached arrays go to `MTAH_WORK_DIR` (set by `scripts/py`), never into the repo.

Rule zero applies to every stage: scripts print aggregates (scores, counts, shapes, timings), never
rows, frames, positions or per-frame labels.
