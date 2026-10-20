# MTAH 2027 agent working agreement

More than a Hack 2027, Phase 1: label every frame of one match `H`/`A`/`D`/`X` (ball possession).
Our work lives on top of the official repository (`upstream` remote). Ranked on `spellF1`, tie-break
`macroF1`. Deadline **2026-11-17**.

## Rule zero: match data never reaches a cloud LLM

The terms forbid uploading, disclosing or making available to any third-party hosted or cloud LLM
any video, dataset, confidential information or extract of them (OVERVIEW.md, "Use of Large Language
Models"). **You, the agent, are a cloud LLM** unless you are explicitly running on a local model.
Read `docs/DATA_POLICY.md` before touching anything near the data. In short:

- Never read, print, `head`, `cat`, grep, plot-to-image, or summarise row-level content of the match
  assets (tracking, homography, Smart Tagging, video, frames) or of frame-level outputs derived from
  them (features, per-frame predictions, our submission CSVs, evaluation reports).
- Write code that reads the data itself, and run it so that it prints **aggregate results only**:
  scores, counts, shapes, timings, error messages without data values. Run it with `scripts/py`,
  which sets the data paths, so the command never names a data path.
- Data lives outside the repo in `~/.mtah2027/data` (`MTAH_DATA_DIR`); intermediate and frame-level
  outputs go to `~/.mtah2027/work` (`MTAH_WORK_DIR`). Never copy them into the repo or into a prompt.
- Optional local tool guards (`.pi/extensions/data-guard.ts`, `.claude/settings.json`, both calling
  `scripts/agent_guard.py`) block calls that name data paths when installed. These configurations
  are ignored, not shipped in clones; the data rules apply with or without them. A block is a stop sign, not a
  puzzle: never work around it (other spelling, globbing, encoding, delegation, codemode). If you
  need information that only the data can give, ask the user, or propose a script that prints an
  aggregate.
- Anything that needs a model to look at the data runs on a **local** model only (Ollama /
  llama.cpp on this Mac). See `docs/DATA_POLICY.md`.
- Public and fine to read: everything committed upstream, including the public 5-minute ground
  truth `evaluation/gt/*_public.csv` and `evaluation/results/baseline_*.csv`.

## Read before working

`docs/SPEC.md` (what we are building and how it is scored), `docs/PLAN.md` (current work),
`docs/DECISIONS.md` (accepted decisions), `docs/CODING.md` (code style and layout),
`docs/DATA_POLICY.md` (data handling). The upstream docs (`CHALLENGE.md`, `DATA.md`, `EVALUATION.md`,
`RULES.md`, `SUBMISSION.md`, `FAQ.md`) are the authority on the competition; where our docs and
theirs disagree, theirs win, and `evaluation/possession_metrics.py` is the metric specification.
Do not claim a capability is implemented or a score improved without evidence.

## Spec-driven loop

1. Select the first ready task from the plan; inspect relevant code and instructions.
2. For changed behavior, update the spec and task acceptance criteria before coding. Resolve routine
   implementation details autonomously. Ask about scope changes, rule questions, or new dependencies.
3. Implement the smallest end-to-end increment that meets the criteria.
4. Verify it; record commands, aggregate results, limitations in the plan.
5. Update task status and affected docs. Record consequential decisions in `docs/DECISIONS.md`,
   including what would justify revisiting them.

Keep documents concise. Never silently turn an assumption into an accepted fact. Tell the user what
each material change does, why, what was verified, and what is still unverified.

## Competition rules that shape the code

- **Labels come from an algorithm.** No hand annotation or hand correction of outputs (RULES.md §1).
- **Only the public 5 minutes of ground truth** may be used. Tune against them knowing they are
  5 minutes and 17 possessions: guard against overfitting (cross-check with the noise floors and
  `--bootstrap` in EVALUATION.md).
- **Reproducible**: one command regenerates the submission file byte-for-byte from the provided
  data. Seed everything; pin dependencies; no manual steps.
- **Public repo, no data in it**: `data/`, results CSVs, reports and any derived frame-level data
  stay out of Git. Check `git status` before every commit.
- Validate every submission with `evaluation/validate_submission.py` before calling it a candidate.

## Multiagent harness

Parallel agent work is authorized when useful. The coordinator owns the plan, integration and final
report; delegate only independent, bounded work. Before delegating, record each assignment in
`docs/PLAN.md` (task, agent, owned paths, completion criteria). One writer per file; shared
interfaces belong to the coordinator. Workers report changed files, checks and aggregate results,
unresolved issues and deviations. Orchestration scratch goes in `.orchestration/` (ignored).
**Every delegated prompt repeats rule zero**; never put data paths or data excerpts in a prompt.
Background delegates and Fusion children may not load this repo's guard, so they rely on the prompt.

## Engineering agreements

- Python only in the project Conda env `./.conda-env` (`scripts/setup-env.sh`); run code with
  `scripts/py`. Never install into base or system Python.
- Ask for confirmation before adding dependencies. Batch proposals with purpose and alternatives.
- Run `scripts/py -m unittest` (or the project test command in `docs/CODING.md`) after code changes
  and report the actual result. Guard tests: `python3 -m unittest discover -s scripts -p 'test_*.py'`.
- Keep the possession logic pure and separate from I/O so it can be unit-tested on synthetic inputs.
- No credentials, data, videos, model weights or frame-level outputs in Git.
