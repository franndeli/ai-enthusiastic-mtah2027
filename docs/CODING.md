# MTAH 2027 — coding guidelines

How any agent or person writes code here. `AGENTS.md` sets process and data rules and wins on
conflict. Upstream files (`games.py`, `starter/`, `evaluation/`) keep upstream style; change them
only when necessary, and never change the scorer.

Build the smallest thing that answers the question, cleanly. Hackathon speed is not a reason for
unclear names, hidden failures or untested rules.

## Principles

1. **Clarity over cleverness.** The next agent has no memory of this session.
2. **Smallest correct change.** No speculative options or "for later" code; delete dead code.
3. **Pure logic separate from I/O.** Possession rules, smoothing, feature computation and spell
   logic are pure functions over arrays, testable on synthetic inputs. Loading and writing are thin
   shells around them.
4. **Make failure explicit, without leaking data.** Validate at boundaries. Errors name the field,
   shape or frame index, never print data values or rows.
5. **Carry meaning in the data.** Frame index (tactical timeline), units (pitch-normalised 0–1 vs
   metres vs pixels), team side (`H`/`A`, never club names in logic), and half travel with values.

## Layout

| Path | Holds |
| --- | --- |
| `possession/` | Our package: pure modules (`features.py`, `rules.py`, `spells.py`, ...) with `test_<module>.py` next to them. Created when the first module is needed. |
| `pipeline/` or `run.py` | The single entry point that regenerates the submission: `scripts/py run.py --out evaluation/results/<name>.csv`. |
| `scripts/` | Repeatable dev tasks (env setup, guard, data status). Not one-off experiments. |
| `docs/` | SPEC, PLAN, DECISIONS, CODING, DATA_POLICY, `research/` notes, the technical document. |
| `config/` | Non-secret settings (`*.json`); machine-specific values in ignored `*.local.json`. |
| upstream files | `games.py`, `starter/`, `evaluation/`: reuse their loaders and the scorer. |

Never in Git: data, video, `~/.mtah2027` contents, `.conda-env/`, submission CSVs other than the
upstream baselines, evaluation reports, model weights, `.orchestration/`.

Do not create `utils`/`helpers`/`misc` grab-bags, empty placeholders, per-agent notes, or `-v2`
copies of files. Version experiments by parameters and a results table, not by copied files.

## Python style

- PEP 8, type hints on public functions, `pathlib`, `logging` in library code (scripts may print
  aggregates), dataclasses for results.
- numpy/pandas vectorised over the 199,319 frames; no per-frame Python loops in hot paths.
- Names include unit or frame base: `frame_idx`, `ball_x_norm`, `speed_m_s`, `window_frames`.
- Determinism: fixed seeds, stable sorts, no reliance on dict/set ordering for output.
- File header: one to three lines stating purpose and constraints. Comments explain why.
- `TODO` only with a plan reference: `# TODO(E1): ...`.

## Tests

- `scripts/py -m unittest discover -s possession -p 'test_*.py'` (stdlib unittest; pytest is a
  dependency proposal).
- Every rule gets synthetic fixtures with edge cases: missing ball, track gaps, identity switches,
  dead-ball stretches, half boundaries, a single-frame flicker.
- Tests never read the gated data.
- Experiments report public-5-minute spellF1 and macroF1 with `--bootstrap` intervals, and possession
  count and mean duration, in `docs/PLAN.md` or `docs/research/`.

## Git

- Small commits, one purpose: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
- Before every commit: `git status`, and check no data, CSV or report is staged.
- Commit code with its tests and the doc updates it requires.

## Done checklist

1. Only what the task asked; no unrelated edits.
2. Pure logic is tested, including failure paths.
3. Tests pass (or the exact failure is reported).
4. Submission candidates pass `validate_submission.py`; the regenerate command reproduces them.
5. No data or frame-level output was printed to an agent or staged in Git.
6. `PLAN.md` records what was verified.
