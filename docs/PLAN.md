# MTAH 2027 — current execution plan

Updated: 2026-10-20

Status vocabulary: `ready`, `in progress`, `blocked`, `done`.
Only mark done when acceptance evidence is recorded. Evidence is aggregate only (DATA_POLICY.md).

| ID | Status | Task | Acceptance criteria |
| --- | --- | --- | --- |
| H0 | done | Working environment and agent harness | Official repo cloned (`upstream`), data root `~/.mtah2027` outside the repo, data guard for pi and Claude Code with passing tests, project Conda env, AGENTS/CLAUDE/docs in place. Evidence: guard tests 12/12 OK; env import check (see H0 notes). |
| H1 | done | Get the match data | User enrols, accepts terms, downloads, and unpacks into `~/.mtah2027/data` keeping the names in `data/README.md`. `scripts/data-status.sh` shows all tactical files `ok`. Evidence 2026-10-22: all nine assets (tactical + broadcast) `ok`; env import check OK; guard tests 12/12. |
| B0 | done | Reproduce the baselines (2026-10-22: `scripts/py starter/baseline.py` byte-identical to the committed files, 1.7 s; public board reproduced) | `scripts/py starter/baseline.py` regenerates the four baselines; `scripts/py evaluation/run_evaluation.py` reproduces the README scoreboard on the public 5 minutes. |
| B1 | ready | Understand the scorer and the public ground truth | From public files only: summarise the 17 public possessions (durations, `D` stretches, hand-overs) and how spellF1 reacts to fragmentation, smoothing and `X`. Notes in `docs/research/`. |
| B2 | ready | Data audit (aggregate only) | A script prints coverage statistics: ball visibility per half, track slot counts, any ownership/alive fields in the ATD, Smart Tagging class counts, homography validity. No rows printed. |
| E1 | blocked on B0 | First method beyond nearest-player smoothing | Proposal with a hypothesis and the measured public-5-min spellF1/macroF1 with bootstrap intervals; reproducible command. |

## S0 — pipeline skeleton (coordinator), 2026-10-22

Status: done. `possession/` (io, features, evidence v1 stub, decoder Viterbi, structure), `run.py`,
`config/possession.json`, contract in `docs/PIPELINE.md`. Evidence: `scripts/py run.py` writes a
valid 199,319-row file in 4.4 s; public spellF1 0.1785 / macroF1 0.4732 (48 possessions; stub
rules, no dead-ball evidence). Dependencies approved and pinned: scikit-learn 1.6.1, scipy 1.15.2,
pyarrow 19.0.1. IDSSE download works via `scripts/fetch_idsse.py` (ndownloader host).

## W1 — build wave, 2026-10-22

Coordinator: main pi session (Orca run, see `.orchestration/`). Every prompt repeats rule zero.
Workers report changed files, test results and aggregate scores only.

| Task | Agent | Owned paths | Completion criteria |
| --- | --- | --- | --- |
| T1 evidence rules | claude | `possession/features.py`, `possession/evidence.py`, their tests | Control/loose-ball, dead-ball, gap-aware and Smart-Tagging evidence implemented as rules with synthetic tests; each clue justified by an ablation table (public spellF1 with bootstrap + full-match possession count/duration); public spellF1 > 0.4242 (best baseline) |
| T2 decoder | codex | `possession/decoder.py`, `possession/structure.py`, their tests, `scripts/sweep.py` | Per-frame transition cost modulation, optional duration-aware variant behind a config flag, synthetic tests incl. half boundaries and 1-frame flicker, switch-cost sweep harness printing a table |
| T3 validation | codex | `scripts/audit_data.py`, `scripts/stress_test.py`, `scripts/score.py` | Aggregate audit (gap statistics per half, nearest-player accuracy and transitions-in-gaps on the public window, Smart-Tagging timing vs public D); gap-transplant stress test with seeds and transition-centred masks; one-line scoring wrapper with bootstrap |
| T4 external data | claude | `possession/external.py`, `scripts/idsse_prepare.py`, `scripts/idsse_eval.py` | IDSSE loaded with dead frames kept, converted to `Match` arrays with H/A/D labels, label-semantics report (aggregate), ATD-like noise injection, leave-one-match-out evaluation of any config with the official metric |

## R0 — research wave (perspective agents), 2026-10-22

Status: done. Four memos delivered and workers released (Orca run `run_d75aa8da3ba7`); synthesis in
`docs/research/phase1_plan.md` (draft, pending team review). Proposed workstreams W0-W8 there
are not yet accepted tasks. No gated data was accessed; no repo file changed by workers.

Review 2026-10-22 (main session, public material and scorer only): the plan's section-3 numbers
reproduce with the official scorer; added measured rows (8-frame lag 0.95, all-D-missed 0.81
spell / 0.59 macro, <2 s spells dropped 0.83), the structural rules for ungraded frames and the
second-half pre-kickoff `D` block, a broadcast-tracking ball-recovery experiment (W7a) ahead of the
video detector, the 3–5 member rule, and the public-data model-artifact decision. Upstream
issues #2 and #8 remain unanswered.

Coordinator: main pi session. Shared brief: `.orchestration/research/BRIEF.md` (ignored).
Workers read public material only (rule zero repeated in every prompt), modify no repo file, and
write exactly one memo each. Completion: memo exists with recommendation, evidence with sources,
risks, effort/compute estimate, and an explicit list of unverified claims.

| Task | Agent | Owned path | Perspective |
| --- | --- | --- | --- |
| R1 | codex | `.orchestration/research/R1_decoder.md` | probabilistic decoder + football possession rules |
| R2 | claude | `.orchestration/research/R2_transfer.md` | ML transfer from external labelled tracking data |
| R3 | codex | `.orchestration/research/R3_validation.md` | validation statistics + red team |
| R4 | claude | `.orchestration/research/R4_vision.md` | computer vision: ball recovery from video |

## H0a — local agent configuration

Status: done. User-requested cleanup of tracked developer-tool settings.
Acceptance: `.pi/` and `.claude/` are ignored and untracked; existing local configurations remain
unchanged; shared rules and reusable guard code/tests remain tracked; docs do not promise guards
on fresh clones. Preserve the existing `.gitignore` edit. Verify Git state and guard tests.
Evidence: `git ls-files .pi .claude` empty; `git check-ignore` confirms both configurations ignored;
`cmp` against HEAD confirms both local files unchanged. `git diff --check` passed;
`python3 -m unittest discover -s scripts -p 'test_*.py'` → 12 tests OK.
No hook code changed; runtime hook behavior was not re-tested. No commit or push performed.

## H0 notes

- Guard tests: `python3 -m unittest discover -s scripts -p 'test_*.py'` → 12 tests OK.
- pi loads `.pi/extensions/data-guard.ts` only after the project is trusted. End-to-end: a pi run with the extension blocked `read` of a dummy `.mp4` with the DATA GUARD message.
- Env: `.conda-env` Python 3.11.17, numpy 2.4.6, pandas 3.0.6, kloppy 3.18.0.
- `scripts/py evaluation/run_evaluation.py` reproduces the README public-5-min scoreboard exactly (smooth125 0.4242 / smooth50 0.3256 / smooth25 0.2868 / raw 0.0378 spellF1).
