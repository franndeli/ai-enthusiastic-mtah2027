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
| E1 | done | First method beyond nearest-player smoothing | v3c, see W1 results: public spellF1 0.6000 [0.386, 0.782] (baseline 0.3256), IDSSE noisy 0.345, `scripts/py run.py`. |
| E2 | ready | Short possessions | We predict 11 of 17 public possessions and 264 on the full match (public rate extrapolates to ~360): find evidence that reacts within ~0.5 s to a clean turnover without flicker; judge on IDSSE clean/noisy spellF1 and on public possession count. |
| E3 | ready | Learned evidence with domain adaptation | Coarsen IDSSE labels (merge spells < ~2 s into the previous owner, D share) towards our granularity, retrain, re-judge (D006 revisit condition). |
| E4 | ready | Technical document polish + team name + public repo push | `docs/TECHNICAL.md` exported as `<team>.pdf`; repo pushed; `git ls-files` audit per DATA_POLICY. |

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
| T6 broadcast ball (W7a) | codex | `possession/broadcast.py`, `scripts/broadcast_align.py`, tests | Formation-based clock alignment; coverage of tactical gap frames by the broadcast ball; go/no-go at >= 15% coverage and < 2 m median disagreement |

### W1 results (2026-10-22)

- **T2 done.** Viterbi with per-frame `cost_scale` and optional age-expanded `min_duration_frames`
  (<= 50); 7 synthetic tests; 0.4 s plain / 2.3 s duration mode per full match. Sweep on the
  interim evidence: costs (12, 3, 12) public spellF1 0.253 [0.155, 0.359]; with
  `min_duration_frames` 2: 0.294 [0.188, 0.400]; 32 public / 531 full-match possessions. The gain
  is inside the noise band, not a selection. Kick-off frames fixed to the FAQ values (11185,
  113742); `games.py` anchors are one frame off in both halves.
- **T3 done (B2 audit).** Full match: ball visible 59.3% (public window 88.2%); 150 missing runs,
  median 10.7 s, p90 33.8 s, max 221 s; 34.7% of graded frames sit in gaps > 10 s; tracked objects
  mean 21.1 (p5 14, p95 23); ball outside bounds on 4.2% of visible frames. Public window:
  nearest-visible-player team accuracy 77.8% (n 5640); last-seen-team persistence on missing
  frames 94.4% (H/A, n 391); 1 of 20 transitions inside a gap. Smart Tagging: 59 set pieces
  (26 throw-ins, 12 corners, 10 goal kicks, 5 free kicks, 6 kick-offs); only 2 of the 4 public D
  stretches overlap a set-piece tag, tag end is ~295 frames after the GT restart (n 2): tags are
  coarse. Stress test on the interim config: native 0.2415, transplant (41% missing, 5 seeds)
  0.230 mean / 0.174 min, transition-centred gaps 5/10/20 s: 0.174 / 0.158 / 0.017.
- **T4 built, blocked then unblocked.** Loader/noise/eval tested on synthetic data (25 tests OK).
  figshare WAF rejects a browser-like User-Agent from scripts; honest UA works. Download running;
  T4b runs the real files.
- **T4b done.** All 7 IDSSE matches load with kloppy (56-63 s each, cached 0.2 s); labels
  lossless (BallPossession only 1/2). Semantics differ from our public window: D share 41-52%
  (ours 19.6% in the first 5 minutes), spells mean 5.5-9.2 s with 25-37% under 2 s (ours 14.2 s
  mean, 35% under 4 s); team swaps 60-67% of transitions (ours 60%). Use IDSSE for robustness and
  relative comparisons, not to set our switching rate.
- **T7 done (accepted).** Ball-free team-shape evidence (attack direction inferred from
  goalkeepers per half; z-scored length/width/speed/drift differences, weights fitted
  self-supervised on clear-control frames of the same match, no ground truth). Public native
  0.4914 -> 0.5935 [0.40, 0.77], macro 0.7724; transplant 0.42 -> 0.54 (min 0.46); tc5/10/20
  0.41 / 0.47 / 0.48; full match 275 possessions, 15.8 s. IDSSE clean 0.377 (unchanged), noisy
  0.259 -> 0.308. 77 tests. Decoder-cost sweep (6/6/6, 3/3/3, 2/4/4, 4/6/3): inert on IDSSE
  (+-0.003) and within noise on public; 6/6/6 kept.
- **Dead ball is the next gap.** Public D recall 0.49 at precision 0.84 (11% predicted vs 20%
  true); same pattern on IDSSE. T8 dispatched.
- **T8 done (accepted).** Dead-ball onset was a median 3.2 s late (IDSSE); added a 3 s speed
  look-ahead and a ball-exit clue. Public D P/R 0.84/0.49 -> 0.87/0.65, spell 0.594 -> 0.613,
  macro 0.772 -> 0.824; IDSSE clean/noisy 0.405/0.332, D F1 0.882/0.852; full-match D share 39%.
  IDSSE's ball is frozen during dead time, so ball-based dead clues cannot be validated there.
- **T9 done (compliance).** Shape weights refit on IDSSE only (supervised, pooled, clean):
  len 0.110, width 1.687, speed -0.535, vel 0.484. Public 0.613 -> 0.600 [0.39, 0.78], macro
  0.827; transplant 0.54/0.43; tc10 0.49 -> 0.42 (17-possession noise); IDSSE clean/noisy
  0.414/0.345 (up). Nothing is fitted on the match any more (reviewed).
- **T5 done: learned evidence NOT adopted** (D006). IDSSE LOMO noisy 0.332 -> 0.436, clean
  0.405 -> 0.547, but on our match 509 possessions / public 0.366 at config costs; equal to the
  rules only after x6 switch costs. Model 2.1 MB, sha256 reproduced 4 times (~7 min training).
- **T10/T11 done.** Independent review (`.orchestration/w1/REVIEW.md`): sanitised data-boundary
  failures, strict frame-id parsing, versioned cache key, one `decode_blocks` entry point shared
  by production and IDSSE evaluators, diagnostic-only sweeps, tests 85 -> 94. Byte-identical
  output before/after; cold-cache and warm-cache runs identical (sha256 374f198f...).
- **Submission candidate `ai-enthusiastic.csv` (v3c)**: valid; public spellF1 0.6000 [0.386,
  0.782], macroF1 0.8271; 11 public possessions (17 true: short spells under-detected); full
  match 264 possessions, 14.6 s mean, 31% D of all frames (39% of graded). Regenerate:
  `scripts/py run.py --out evaluation/results/ai-enthusiastic.csv`.
- **T6 done: W7a NO-GO.** 31 of 253 broadcast runs aligned (602 s), 313 of 64,643 tactical gap
  frames recovered (0.48%); both-visible ball disagreement median 2.6 m, p90 9.6 m; player residual
  after alignment 0.8 m median (11.9 m before). Conservative aligner; a looser one is not worth
  today's time. Broadcast assets dropped for Phase 1 unless revisited.

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
