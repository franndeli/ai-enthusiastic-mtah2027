# MTAH 2027 — current execution plan

Updated: 2026-10-28

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
| E2 | done | Short possessions | W2: last-touch ownership (v4). Metrica sample clean/noisy 0.461/0.439 -> 0.563/0.528, IDSSE 0.414/0.344 -> 0.508/0.401, every match up; public 0.600 -> 0.525 [0.35, 0.70] (15 of 17 possessions; 0.600 with oracle dead frames), full match 314 possessions. `docs/research/last_touch_ownership.md`, D011. |
| E6 | done | Touch precision and restart clues (v5) | W3: pass-by touch filter, touch-now evidence, 3-frame touches, out-of-pitch player and slow-far-ball dead clues. Metrica clean/noisy 0.563/0.518 -> 0.605/0.570, IDSSE 0.508/0.397 -> 0.576/0.435, every match and seed up; public 0.525 -> 0.533 [0.36, 0.69], macro 0.823; 305 full-match possessions. `docs/research/touch_precision.md`, D014. |
| E5 | blocked | Dead-ball stretches with a tracked, slow ball | On the public D stretches the ATD keeps reporting a ball moving at walking pace ~3 m from the players (seen 66% vs 10-22% on Metrica dead frames); v5 still misses two of the four. W3 measured the regime (`docs/research/tracked_ball_stoppages.md`): the conjunction clues with no alive false positives (`w_out_player`, `w_slow_far` adopted; `w_quiet`, `w_edge` implemented, off) cover at most a tenth of those frames, the broad "slow ball + walking players" clue is wrong on 3-7% of alive frames on every judge, and Metrica's seen-ball dead frames are label slop. Blocked on a judge that contains the regime (corrected same-provider tracking, or an externally trained video ball detector). |
| E3 | done (not adopted) | Learned evidence with domain adaptation | V2 tested fixed <2 s external ownership-label coarsening with team-only learning: reserved IDSSE noisy +0.04772, but target/stress regressions; rejected. See `docs/research/factorized_generalization.md`. |
| E4 | ready | Technical document polish + team name + public repo push | `docs/TECHNICAL.md` exported as `<team>.pdf`; repo pushed; `git ls-files` audit per DATA_POLICY. |

## W3 — touch precision and restart clues (main session), 2026-10-28

Status: done, **v5 adopted** (D014). Single-agent wave after a review of the organisers'
repository (issues #2–#17, closed PRs 1/7/9/12/13: rules clarifications, a colleague team's HMM
with sharp speed + edge dead cues at public 0.607). Guard active throughout; aggregates only.
Full evidence and the rejected ideas: `docs/research/touch_precision.md`.

- Diagnosis on the Metrica judge: possessions under 4 s (45 % of all) have recall 1–40 %; the
  oracle-team score is 0.81 against 0.56 (oracle-dead 0.66), and re-weighting any existing clue
  moves the externals by ≤ 0.01. Touch votes whose ball never changes velocity (11 %) are at
  coin-flip precision; the rest at 84 %.
- v5 = v4 + pass-by filter (`touch_passby_dv_m_s` 2.0 with a label-free jitter floor), touch-now
  evidence (`w_touch_now` 2.0), `touch_min_frames` 3, `w_out_player` 1.0, `w_slow_far` 1.0.
- Results: Metrica clean/noisy 0.563/0.518 -> 0.605/0.570, IDSSE 0.508/0.397 -> 0.576/0.435,
  all ten matches and both seeds up; public 0.533 [0.36, 0.69] (v4 0.525), macro 0.823,
  16/17 possessions; transplant 0.496/0.400; tc 0.317/0.415/0.400; 137 tests OK; validator OK;
  cold/warm identical (sha256 `cfb99407f6abb4f1`); `scripts/verify_submission.py` added.
- New diagnostics: `scripts/short_spell_stats.py`, `touch_precision_stats.py`,
  `restart_clue_stats.py`, `speed_domain_stats.py` (the target's player speeds never fall below
  0.5 m/s the way external tracking does: the slow-share clue does not transfer, mean speed does).
- Not resolved: E5 (stoppages with a tracked slow ball) remains the public window's main loss;
  `w_slow_far` touches it only lightly (public D stretches: 6–12 % coverage).
- E5 follow-up (same session): the target's tracked-ball stoppages measured against Metrica
  (`scripts/tracked_dead_stats.py`); `w_quiet` (unattended slow ball + walking players) and a
  jitter-robust ball-motion clue implemented and measured: no public gain, Metrica -0.001/-0.008;
  `w_quiet` kept off. `docs/research/tracked_ball_stoppages.md`; E5 marked blocked.


## W2 — last-touch ownership, provider-matched judge (main session), 2026-10-26

Status: done, **v4 adopted** (D011). Single-agent wave; no data path or match content entered
the context (guard active throughout, aggregates only). Full evidence and rejected ideas:
`docs/research/last_touch_ownership.md`.

- Read the public ground truth: six of 17 possessions are under 4 s (0.76 s and 1.32 s exist);
  ownership is "last team to play the ball". v3c's sustained-control model cannot produce them.
- New external judge with the competition's own provider: three Metrica sample matches
  (`scripts/fetch_metrica.py`, `possession/metrica.py`, `scripts/metrica_eval.py`), labels
  derived from synchronised events (tested synthetically), spell statistics close to the public
  window's (median 5.3 s vs 7.6 s; < 4 s 40-44% vs 35%); ball missing 32-41% natively.
- v4 = v3c + causal touch memory (`w_touch` 1, 20 s decay, 5 s extra decay while the ball is
  unseen), gap memory off, ball-travel alive clue (`w_moving` 1.5 at 5 m/s), costs 16/8/8.
- Results: Metrica clean/noisy 0.563/0.528 (v3c 0.461/0.439), IDSSE 0.508/0.401 (0.414/0.344),
  all ten matches and both seeds up; public 0.525 [0.35, 0.70] (v3c 0.600 [0.39, 0.78]),
  macro 0.811, 15/17 possessions, 0.600 with oracle D frames; transplant 10 seeds 0.49/0.40
  (v3c 0.56/0.43); full match 314 possessions of 12.7 s; 108 tests OK; validator OK; cold and
  warm generation identical (sha256 `1f69e8429b62f25d`).
- Not resolved: dead-ball stretches where the ATD tracks a slow ball (task E5).

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

## V1 — OpenAI-only generalization review

Status: done — no candidate accepted. Coordinator owns integration, acceptance and this plan.
v3c production files remained unchanged; experiments live in ignored `.orchestration/v1/`.
No new dependencies. Full evidence: `docs/research/generalization_review.md`.
Acceptance: SPEC generalization gate; all attempts reported, no adoption from public-window tuning.
Use whole-match external evaluation and seeded paired noise; disclose prior IDSSE reuse and leakage.
Workers may edit only their listed scratch paths, never production code/config or each other's files.

| Task | Agent/model requested | Owned paths | Completion criteria |
| --- | --- | --- | --- |
| V1-A done: validation audit | OpenAI Codex / gpt-6.1-sol high | `.orchestration/v1/audit/` | Audit leakage, scorer and fit/evaluation boundaries; recommend executable strict gate; independently challenge claimed generalization. Code/public docs only, aggregate outputs. |
| V1-B done: no finalist | OpenAI Codex / gpt-6-astra high | `.orchestration/v1/rules/` | Predeclare <= 4 evidence/control hypotheses before scoring; external-only development; at most one frozen finalist, synthetic tests and aggregate comparisons. No production edits. |
| V1-C done: no finalist | OpenAI Codex / gpt-6-sol high | `.orchestration/v1/decoder/` | Predeclare <= 4 decoder/temporal hypotheses before scoring; external-only development; at most one frozen finalist, synthetic tests and aggregate comparisons. No production edits. |

Coordinator reserves the final public/stress evaluation and production changes. A hypothesis must
not be adjusted after confirmation results. Workers use the first four sorted external matches
for development only; remaining three are coordinator confirmation (historically reused, not unseen).
All fitting for claimed held-out scores must exclude evaluation matches. Rule zero repeated in
all dispatches. Each report includes commands, all candidates, failures and limitations.

Evidence: Orca run `run_5871893e60b7`, three distinct requested/effective OpenAI models confirmed;
all workers completed and were released. Seven predeclared alternatives tested; coordinator
independently reran both evaluators. Best rule noisy-development gain +0.01635; best decoder
+0.00479; both below +0.03. No alternative advanced to confirmation or target scoring; no
parameter revisions after results. Development scores use historically pooled shape coefficients
and are exploratory, not held-out generalization evidence. Audit found shape-fit overlap and
shared-temperature leakage in the unused learned experiment; TECHNICAL now qualifies past claims.
Three reused confirmation matches cannot establish strong new-match evidence; no weaker
statistical gate was substituted.

Coordinator reproduced v3c public spellF1 0.6000 [0.3857, 0.7823] (90% interval), macroF1
0.82715; 10-seed transplant mean/min 0.55919/0.42963, transition 5/10/20 s 0.40000/0.41538/0.47692.
Official validator exit 0; two generations SHA-256 identical (`374f198f07ccecbc...`).
Development-only shape-fit comparator was separately run on the three confirmation matches
(clean and seeds 101/307/911); no candidate comparison or production change implied.
Checks: 94 package + 12 guard + 6 audit + 4 decoder + 8 rule synthetic tests passed.
Production diff empty. Only documentation updated; no commit/push. Further credible promotion
needs leakage-free fitting/calibration and more untouched, comparable external-match evidence.

## V2 — substantive learning and validation repair

Status: done — broader experiments completed, no production promotion. Continued after the
user's request to push beyond V1. Production v3c remained frozen; no dependencies or target-data fitting. Coordinator owns docs, common evaluation
and acceptance. V1's numerical/regression gates remain in force; do not use the public window
to tune a failed candidate. This is a broader experiment wave, not a claim of improvement.

| Task | OpenAI model | Owned paths | Completion criteria |
| --- | --- | --- | --- |
| V2-A learned stoppages | gpt-6-astra high | `.orchestration/v2/stoppages/` | Fixed player-only dead/alive learner, avoiding IDSSE's frozen-ball shortcut; outer-match-exclusive training, synthetic tests, clean/noisy development scores; at most two predeclared recipes, no target scoring. |
| V2-B domain-adapted ownership | gpt-6-sol high | `.orchestration/v2/ownership/` | Fixed external-label coarsening and learned team-only evidence while preserving rule dead evidence; all fit/calibration excludes evaluation matches, synthetic tests, clean/noisy development scores; at most two predeclared recipes, no target scoring. |
| V2-C untouched validation data | gpt-6.1-sol high | `.orchestration/v2/validation/` | Public-source audit for independent possession-labelled tracking with license/access/semantics evidence; do not treat event-derived proxy labels as true frame possession; also review common evaluation interface without viewing candidate results. |
| V2-D fixed factorized combination | coordinator | `.orchestration/v2/combined/` | Predeclare one composition of the two 50% blends before either blended recipe's results; same outer-fold fit boundaries and gate, no blend search or target tuning. |
| V2-E new-provider weak corroboration | gpt-6.1-sol high, reused validator | `.orchestration/v2/skillcorner/` | Download only first three sorted pinned SkillCorner bundles (~266 MB), verify hashes, pure synthetic-tested 10→25 Hz adapter; score frozen ownership recipe without fitting on these matches; report provider-label caveats, all aggregate coverage and scores. No production changes. |

First-four sorted external matches remain development; candidates screened with four-fold
cross-fitting, including shape coefficients fitted only to each outer training set. Paired noisy
seeds 11/29, fixed recipe selection on external development only. Any frozen finalist must be
independently rerun on the remaining three matches with all fitted components trained only on
the first four, then judged against the untouched-production target baseline once. Confirmation
remains historically reused and inadequate for an unconditional new-match claim; acquisition of
meaningfully independent evidence is a separate task, not a reason to manufacture significance.
Scratch-only code/artifacts until acceptance; row-level material remains outside the repository.
Rule zero repeated in every dispatch; workers must report failed experiments too.

### V2 results

Full evidence and exact commands: `docs/research/factorized_generalization.md`. Run
`run_8d721e3b9383`, three distinct OpenAI models and four completed dispatches (validator reused
for acquisition). Five recipes: pure/half-blended stoppages, pure/half-blended ownership, fixed
both-half composition. All fitting excluded the evaluated match, including baseline shape
weights; no calibration-label leakage. Shared evaluator independently audited and hardened.

- **V2-A rejected:** player-only stoppages noisy spell +0.01025 despite dead F1 +0.04194;
  clean-match regression. Half-blended stoppages regress.
- **V2-B selected externally, rejected for deployment:** coarsened ownership dev clean/noisy
  gains +0.06268/+0.07456, positive on all four outer folds. Frozen reserved-three confirmation
  gains +0.03598/+0.04772, positive on each match. Both reproduced independently by coordinator.
  Target public spell 0.60000→0.55484, macro 0.82715→0.80783; transplant 0.55919→0.51884;
  tc5/10/20 all regress. No target-driven revision or runner-up target scoring.
- **V2-D passes development screen but not selected:** fixed combination noisy +0.04916, lower
  than ownership-only; half-blended ownership +0.05227. No further target tests.
- **V2-C/E completed:** found and hash-verified three new SkillCorner matches (nine assets,
  266,007,186 bytes), no new dependencies; synthetic-tested local 10→25 Hz adapter. Frozen
  ownership weak-label spell 0.47945→0.49111 (+0.01166), one slight match regression;
  coordinator rerun reproduced. Provider phase/alive labels are algorithmic proxies, not gold,
  so this does not rescue failed promotion. Remaining corpus matches not acquired/evaluated.
- **Checks:** 156 tests passed (94 package, 12 guards, 50 V2 synthetic). Official validator exit
  0 for rejected artifact and frozen production. Cold/warm production generations match original
  SHA-256 `374f198f07ccecbc...`; production diff empty. All dispatches settled; ownership and
  validation/acquisition workers released, stoppage worker preserved by Orca as user-owned.
  No commit or push. Reused IDSSE/small-sample uncertainty and source transfer remain limitations.

## V3 — observation-model research, RunPod compute

Status: done — no production promotion. User requests a meaningful improvement beyond public spellF1 0.600, forbids
heavy local CPU/GPU work, and authorizes RunPod and agents. Preserve all pre-existing V1/V2 work
and the v3c candidate. Coordinator owns integration, external experiment execution, target checks,
resource lifecycle and this plan. No target fitting/manual annotations/new dependencies.

| Task | Agent | Owned paths | Completion criteria |
| --- | --- | --- | --- |
| V3-A contact/trajectory evidence | supervised Claude worker (Codex startup blocked) | `.orchestration/v3/contact/` | Review prior failures; predeclare <=2 materially different physics/contact recipes; pure candidate code and synthetic tests only; no real-match jobs, target scoring, production edits or dependency additions. Report mechanism, risks and exact evaluator interface. |
| V3-B transfer audit and temporal ownership | supervised Claude worker | `.orchestration/v3/temporal/` | Audit V2 transfer bottlenecks from code/aggregates; implement <=2 externally trainable, source-robust temporal ownership recipes using existing dependencies; declare before scores, synthetic checks only, whole-match fit exclusions, no target use. |
| V3-C remote evaluation and acceptance | coordinator | shared V3 evaluator/remote scripts, docs; accepted production changes only | Bounded RunPod execution on public external data only; all recipes/results recorded, unchanged SPEC gate, official validator and reproducibility for any finalist; verify no billable compute remains. |
| V3-D fixed source-balanced ownership | reused V3-B Claude worker | `.orchestration/v3/multisource/` | Before any V3 scores, declare exactly one equal-provider training variant of the frozen gated temporal design using only the existing three SkillCorner development matches plus supplied IDSSE train IDs. Weak ownership labels only, fixed recipe, no evaluation feedback, synthetic tests. |
| V3-E evaluation/source review | reused V3-A Claude worker | `.orchestration/v3/audit/`; factual corrections only to contact memos | Review shared external evaluator and temporal candidate boundaries without results/target data; synthetic tests for leakage and immutability; correct overstated baseline-window claims. No candidate implementation or threshold changes. |
| V3-F spatial gap ownership | coordinator | `.orchestration/v3/gap_spatial/`, `.orchestration/v3/composed/` | Gap-only classifier using team geometry around last/next/interpolated ball anchors plus shape/memory; masked external training, equal-provider weighting, never target labels. Two fixed recipes: standalone gap correction and its visibility-weighted composition with frozen contact_filter (composition declared after contact's external screen, before gap results or any target check). No blend search or revisions after scores; synthetic tests, same external screen. |

The initial remote budget is capped at $3 and 3 wall-clock hours for one inexpensive instance;
inspect price before launch and terminate on completion/failure. Expand only if justified and
confirmed. Workers do not launch compute or access credentials. Existing three SkillCorner
matches are development-only/reused, never a fresh holdout; reserve any newly acquired matches
before inspection. Native target checks remain coordinator-only and follow external selection.
Reserved before any V3 scores: six next-sorted, previously unacquired SkillCorner IDs
1925299/1927964/1953632/1959846/1986691/1996435, pinned commit
`4340d274572876239c154c90bc507a9b3250a656`. These are confirmation-only weak-label corroboration,
not gold, and will only be acquired for a frozen externally selected finalist. The existing first
three may support development/training, never pristine confirmation.

Early verification: six new synthetic evaluator-boundary checks passed; 94 package + 12 guard
tests passed. Cold/warm frozen production regenerated identically with original SHA-256 and
validator exit 0 (8.6 s total local verification; one thread). RunPod first provisioning attempt
was terminated (delete exit 0, owned inventory empty) after an SSH response-schema mismatch;
retry fixes the schema, not a model recipe. Codex startup failed at an update prompt; its supervised
retry uses Claude, and the failed owned terminal was released by Orca. An earlier launcher task
`97611595-ee5f-4564-a234-e2b745c21dad` was rejected at 18:23 UTC with `consumer_fenced`
(coordinator not bound to the run). Its delayed notification was reconciled after closeout; later
supervised launches supplied the completed work. No job was restarted; owned-pod inventory was
rechecked empty.

### V3 development checkpoint (before confirmation/target)

Seven recipes measured on RunPod, four-fold external fitting, paired seeds 11/29. Mean noisy
spellF1 deltas: contact_filter **+0.04079**, deflection_flight +0.01253, residual_rule +0.02785,
gated_learned -0.00389, source_balanced +0.00086, spatial_gap +0.01869,
**contact_spatial +0.06396**. The fixed composition has clean gain +0.07287 and noisy macro gain
+0.01341, with clean/noisy gains on every development match. Select **contact_spatial only** for
confirmation, before any target evaluation; no runner-up target test if it fails. Composition
source SHA-256 `1fdf7f58188bdcdee229c768ce7a5cabf081ad979689f93d5447e8168fcbaf6d`.
This is a measured external improvement, not yet a better submission or proof of target transfer.

Independent source/mock audit found incomplete input-mutation checks and missing D-column
verification in the experimental evaluator, not a model bug. Coordinator hardened both; 36 audit
checks pass without expected failures. Contact's original gains reproduced exactly with the
hardened evaluator; the composition was first evaluated with it. Team symmetry is covered by each
candidate's synthetic tests, not asserted from performance. RunPod smoke checks loaded all seven
public caches and used matching pinned package versions. Three development SkillCorner bundles
were fetched/hash-verified on RunPod, never sent to an LLM; six reserved bundles remain untouched
pending the frozen finalist's confirmation.

### V3 final decision

Frozen composition passed reserved IDSSE numerical confirmation: clean +0.03616, noisy +0.04314,
all three matches positive, descriptive noisy paired interval [0.02403, 0.05608]; reused-match and
small-sample caveats remain (sign-flip p=0.25). Six newly acquired SkillCorner matches gave mixed
weak-label corroboration: mean +0.00751, three positive/three negative, interval [-0.01075, 0.02400].
No provider or recipe selection followed those outcomes.

Target final check **rejects**: public spellF1 0.60000→0.46061, macroF1 0.82715→0.78172;
transplant mean 0.55919→0.45403; transition 5/10/20 s 0.29677/0.35862/0.34286 versus
0.40000/0.41538/0.47692. Predicted public possessions 11→16, full-match 264→329; correct counts
are not correct intervals. Candidate validator exit 0 and independent reload/regeneration was
byte-identical (`4aa15b6a7104daa8...`), but it is not an accepted submission. No runner-up target
score or parameter revision. Local inference was bounded to one numerical thread, native 12.0 s
and stress 29.2 s; training and all external evaluations ran on RunPod. First-half stress inference
has a synthetic bit-equivalence check and reproduces every baseline stress score exactly.

## V4 — previously unused automatic phase ownership

Status: done — both fixed recipes fail external development; no target check or promotion.
Distinct information source, not a rescue/retuning of rejected V3 recipes.
All V3 implementations and output settings are frozen. Pre-score V4 review retains v3c's team
confidence cap (tags share tracking inputs) and corrects duplicate-vs-conflict normalization.
Same coordinator, same bounded remote
lease; competition assets remain local and opaque to agents. No dependency additions.

| Task | Agent | Owned paths | Completion criteria |
| --- | --- | --- | --- |
| V4-A phase observation model | supervised Claude worker | `.orchestration/v4/phases/` | At most two predeclared soft ownership-prior recipes from publicly described automatic phase codes; pure logic, synthetic tests, no real-match runs or target scoring. No target labels/pseudo-label fitting. |
| V4-B external event observation adapter | supervised Claude worker | `.orchestration/v4/events/` | Genuine IDSSE public event observations independently aligned to cached tracking; no evaluated ownership labels used to construct inputs. Source/schema-based alignment, synthetic tests, safe aggregate audit/fetch command for RunPod. Stop if correct timebase cannot be established. |
| V4-C evaluation/integration | coordinator | V4 shared evaluator, docs, accepted production changes only | Freeze external event-to-phase semantics and degraded-observation recipe before scoring; same outer fitting/paired gate, one frozen target finalist. Explicitly disclose event-vs-automatic-phase domain gap. |

V4 source-readiness checkpoint (before results): all seven real event feeds parsed on RunPod. Independent supplied
CalculatedFrame/CalculatedTimestamp metadata verify clock conversion on 180 events (one <=1-frame
residual, others numerical precision). Original EventTime annotations are nevertheless noisy;
pass-speed profiles show 2–3 s displacement on five matches, consistent with the source paper's
explicit warning. No offset is tuned or corrected. Evaluate the original observations as a noisy
coarse auxiliary feed, not as precise touch timestamps. Code/test review: 19 phase + 23 adapter +
12 coordinator synthetic checks pass; production 94 and guards 12 still pass. At that checkpoint candidate scoring
had not yet begun. Final results follow; no V4 target outputs were created.

### V4 final decision and session closeout

External noisy spellF1: baseline 0.30732; phase_gap 0.32288 (+0.01556), phase_uncertain 0.32797
(+0.02065). Clean gains 0.00000/+0.01888. All four noisy match deltas positive, but neither reaches
+0.03: **select none**, no reserved-three comparison, no target score or strength/taper/gate rescue.
Evidence and timestamp limitations: `docs/research/automatic_phase_evidence.md`. The nine measured
V3/V4 recipes have not produced an accepted improvement over public spellF1 0.6000.

Verification: 265 tests passed (94 production, 12 guard, 105 V3, 54 V4). Production code/config and
pins have no diff; pre-existing V1/V2 documentation changes preserved. Final single-thread
`scripts/py .orchestration/v2/check_production.py` rechecked cold/warm regeneration: both match
`374f198f07ccecbc5f7b6231235c40eb6bbda3c66c391687bff6de5dc22d7758`, official validator exit 0.
Canonical package and guard unittest commands plus discovery in each V3/V4 scratch suite passed;
`git diff --check` passed, nothing staged, scratch confirmed ignored. No commit or push. Failed
artifacts remain isolated, never replacing the v3c submission.

RunPod closed: pod f4ykdqal54ajdo deletion exit 0, then independent owned-pod inventory empty;
initial failed provisioning pod snk3ilflbn403o was also verified deleted. Successful lease lasted
2.56 h at API-reported $0.24/h, approximately **$0.61 compute**; this is a runtime/rate estimate,
not a reconciled invoice (brief initial provisioning/storage charges not established). V4 workers
and the owned V3 temporal/audit worker released; the user-owned/external contact terminal retained
by Orca was left untouched. Managed lease task `143bb4f7-c26b-401e-aeae-a740f849e86f`, visible
terminal `term_aa02b04f-c5a4-4ec6-a22c-e4c95ff4c685`; coordinator run `run_d89006b3db08`.

## H0 notes

- Guard tests: `python3 -m unittest discover -s scripts -p 'test_*.py'` → 12 tests OK.
- pi loads `.pi/extensions/data-guard.ts` only after the project is trusted. End-to-end: a pi run with the extension blocked `read` of a dummy `.mp4` with the DATA GUARD message.
- Env: `.conda-env` Python 3.11.17, numpy 2.4.6, pandas 3.0.6, kloppy 3.18.0.
- `scripts/py evaluation/run_evaluation.py` reproduces the README public-5-min scoreboard exactly (smooth125 0.4242 / smooth50 0.3256 / smooth25 0.2868 / raw 0.0378 spellF1).
