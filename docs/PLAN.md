# MTAH 2027 — current execution plan

Updated: 2026-10-25

Status vocabulary: `ready`, `in progress`, `blocked`, `done`.
Only mark done when acceptance evidence is recorded. Evidence is aggregate only (DATA_POLICY.md).

| ID | Status | Task | Acceptance criteria |
| --- | --- | --- | --- |
| H0 | done | Working environment and agent harness | Official repo cloned (`upstream`), data root `~/.mtah2027` outside the repo, data guard for pi and Claude Code with passing tests, project Conda env, AGENTS/CLAUDE/docs in place. Evidence: guard tests 12/12 OK; env import check (see H0 notes). |
| H1 | blocked on user | Get the match data | User enrols, accepts terms, downloads, and unpacks into `~/.mtah2027/data` keeping the names in `data/README.md`. `scripts/data-status.sh` shows all tactical files `ok`. |
| B0 | blocked on H1 | Reproduce the baselines | `scripts/py starter/baseline.py` regenerates the four baselines; `scripts/py evaluation/run_evaluation.py` reproduces the README scoreboard on the public 5 minutes. |
| B1 | ready | Understand the scorer and the public ground truth | From public files only: summarise the 17 public possessions (durations, `D` stretches, hand-overs) and how spellF1 reacts to fragmentation, smoothing and `X`. Notes in `docs/research/`. |
| B2 | blocked on H1 | Data audit (aggregate only) | A script prints coverage statistics: ball visibility per half, track slot counts, any ownership/alive fields in the ATD, Smart Tagging class counts, homography validity. No rows printed. |
| E1 | done (external) | First method beyond nearest-player smoothing: v3 (`possession/`, `run.py`) | Hypothesis: two-stage Viterbi with learned dead-ball and control/shape evidence beats window voting. Evidence: leave-one-game-out on Metrica's public sample games, spellF1 0.544 vs 0.247 for smooth50, paired Δ +0.298 [+0.286, +0.312] (`docs/research/v3-external-validation.md`). Tests: `python -m unittest discover -s possession -p 'test_*.py'` → 5 OK. |
| E2 | ready (local only) | Run v3 on the match | `python run.py --out evaluation/results/check50.csv --baseline 50` matches `baseline_smooth50.csv` (`scripts/compare_csv.py`); `python run.py --out evaluation/results/v3.csv` passes the validator; public-5-min scoreboard with `--bootstrap 1000` recorded as a sanity check only; full-match aggregates (possessions, D share) recorded. Two runs give the same SHA-256. |
| U1 | ready | Sync upstream (23 Oct rule clarifications) | `git fetch upstream && git merge upstream/main`. Upstream now says the public 5 minutes are not for training or tuning and that external data must be about other games. AGENTS.md "Tune against them" must be aligned with that. |
| X1 | ready | Second external set: Sportec Open DFL (7 matches, per-frame BallPossession/BallStatus, CC-BY) | Download locally (figshare), add a loader to `possession/external.py`, rerun `scripts/validate_external.py`. |

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
