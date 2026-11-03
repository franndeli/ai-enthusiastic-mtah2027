# Colleague branch review

## Verdict and scope

**Do not merge either branch wholesale. Keep both as research contributions.** v5 is a working,
reproducible alternative, with a particularly useful dead-ball hypothesis. Route A has a useful
feature/ablation architecture, but its default tag calibration uses the competition ground truth
and must be removed/replaced before submission under the current rules. Neither branch has
established an apples-to-apples improvement over main on independent external matches.

Reviewed snapshots (branch-local paths/line numbers below refer to these commits):

| Source | Commit | Role |
|---|---|---|
| `origin/feat/possession-pipeline-v5` | `7db538a080f7460fcd7dc3794a67489a87518b86` | Colleague's two-head logistic + two-stage Viterbi |
| `origin/feat/route-a-possession` | `baf223bced0bc16fc6be69749d94a8380ba32b5c` | Colleague's MLP + HMM-style Viterbi + tags |
| `main` | `60d0bfcdf62cd1141efe26128d5fe01b5db30bf8` | Current v4 reference |
| `upstream/main` | `f8f2629ff1e0985f81d7e59d8413678514ee1530` | Latest organiser rules checked during review |

Both colleague branches fork at `6215ecb`, before main's pipeline. Their version numbers do not
represent successive versions of main. No merge, branch edit, dependency installation, heavy
training, remote compute or model promotion was performed. Match assets and frame-level results
stayed local and outside the repository; only aggregates entered the review.

## What I independently verified

All tests passed: **v5 13, Route A 79, main 108, guards 12**. Passing tests do not cover the
additional defects below. v5 and main were each generated twice through their actual CLI in the
project environment with one numerical thread; both produced identical bytes and passed the
official submission validator. The official scorer is unchanged between main and latest upstream.

| Aggregate check | Main v4 | Colleague v5 |
|---|---:|---:|
| Submission rows | 199,319 | 199,319 |
| Public spellF1 | 0.52500 | 0.57333 |
| Public spellF1 90% bootstrap interval | [0.350, 0.700] | [0.360, 0.773] |
| Public macroF1 | 0.81118 | 0.84924 |
| Public predicted possessions (truth: 17) | 15 | 13 |
| Full-match predicted possessions | 314 | 221 |
| Full-match predicted dead share, graded frames | 37.33% | 51.54% |
| Public dead precision / recall | 95.03% / 59.84% | 76.72% / 81.42% |
| 10-seed gap transplant spellF1 mean / min | 0.49196 / 0.40000 | 0.47447 / 0.37419 |
| Transition-centred mask, 5 s | 0.34074 | 0.44444 |
| Transition-centred mask, 10 s | 0.41538 | 0.38400 |
| Transition-centred mask, 20 s | 0.40000 | 0.35385 |

These are correctness/stress checks, **not a public-window model-selection exercise**. No
parameters were changed after any result. The masks are exactly main's existing
`scripts/stress_test.py`, seeds 0–9 and durations 5/10/20 s. v5's stress adapter reproduced its
native CLI predictions exactly before applying masks. Stress inference for both took 38.3 s.
The public advantage does not consistently survive missingness; it is not evidence of a private
score gain. Predicted counts/dead shares are not estimates of the private ground truth.

Additional verification:

- v5's reader + baseline mode regenerates upstream `baseline_smooth50.csv` **byte-for-byte**.
  Its target frame mapping and team mapping are therefore supported, not merely assumed.
- Main regeneration SHA-256: `cf06b962d6f8dc1ac1900bf120fa81338bcec301eb881bec9aa924309daed3a7`.
- v5 regeneration SHA-256: `eee09a1829f8146f3c8f9c2332b622e725b0dfd117fc6cb7673eae6cc7e667f7`.
- Route A's required `model.npz` is not present locally and no trained model is shipped on its
  branch. Its final prediction, reported public 0.518, full-match 371 possessions and external
  0.559 are **branch-reported, not independently reproduced here**. Training is documented, so
  absence of the artifact is a verification limitation, not proof that reproduction is impossible.
- External training/evaluation tables in either branch were source-audited, not rerun. No claim
  here that the seven/20-match means or model-training hashes independently reproduced.

## Findings requiring action

### 1. Route A: public-label fitting is on the default inference path — rules blocker

**Locations:** `run.py:39–59`; `possession/smart_tagging.py:113–164`;
`possession/train.py:30–36`; `docs/research/improvement_round.md`, I4.

`game_log_evidence()` loads public GT, calls `calibrate()` on those labels, and supplies the
result to every inference run. The default tag weight is 0.25. This learns the dead prior,
phase agreement, restart dead probability and restart team agreement from the evaluation match.
The research note also says the tag weight was selected with a two-fold split of the public
window. Calling this "calibration" rather than training does not make it held-out evaluation.
A synthetic probe changes only the evaluation labels and changes all 100 tag-evidence rows.

Latest upstream `RULES.md` §2 and `FAQ.md` explicitly forbid training **or tuning** on the public
five minutes. The branch's original interpretation predates that clarification. Its public
number is acknowledged as in-sample in its own research note; it cannot validate the tag gain.
The dropout `DEAD_RATIO=5` also comes from a label-conditioned public-window measurement, and
count-based selection toward an extrapolated ~360 possessions needs the same provenance audit.

**Action:** remove the public-GT dependency from production; set tag evidence aside until its
calibration/selection is external-only. `--smart-tagging-weight 0` bypasses the runtime tag fit,
but alone does not undo the other target-informed choices. Do not reuse the published default
score as an independent measurement of a corrected model.

### 2. v5: the headline score answers a modified-label problem — evidence blocker

**Locations:** `possession/dfl.py:85–111`; `scripts/validate_dfl.py:74–105`;
`scripts/train_model.py:119–135`; `docs/research/v5-dfl.md`.

The branch removes opponent ownership runs shorter than 2 s, then trains **and scores** on that
coarsened DFL truth. Its explanation treats "controlled" as requiring this duration. However,
CHALLENGE explicitly says its verbal proxies are not a technical specification and the ground
truth is not smoothed. It contains legitimate sub-two-second possessions.

Public-only semantic probe: applying the branch's `controlled()` to otherwise perfect public
labels changes 52 frames, reduces 17 possessions to 13, and scores **0.82667 rather than 1.0**.
This is not a measurement of v5 predictions; it disproves the assertion that this coarsening is
necessarily equivalent to the task. External-label coarsening can still be a valid *training
hypothesis*, but it should not silently redefine the primary evaluation target.

The branch itself reports that a tested variant's DFL gain disappears against raw labels. Its
0.598 DFL score therefore cannot be compared with main's raw-label IDSSE score or Route A's
0.559. Metrica event-to-label conventions also differ between branches. All three require the
same adapters, untouched evaluation labels and noise masks before ranking them.

### 3. v5: new-match uncertainty is understated

**Locations:** `possession/reference.py:104–136`; `scripts/validate_dfl.py:47–57`;
`scripts/validate_external.py`; `PIPELINE.md`, "both significant" claim.

`paired_bootstrap()` independently resamples possessions within each game/noise case, then pools
counts. It does not resample whole matches or preserve shared possessions across noise seeds.
The interval describes a conditional within-corpus perturbation, not transfer to a new match.
Its pooled statistic also differs from the displayed equal-weight mean of match scores.

Synthetic demonstration: repeating the *identical* case nine times shrinks interval width from
**0.29485 to 0.09632** with unchanged point estimate and no additional football evidence.
The same-match seeds are repeated measurements, not extra independent matches. Even the
20-match SkillCorner interval is possession-level, not a match-level transfer interval.

**Action:** average seeds within each match, report every paired match delta, and use match-level
intervals. Selection optimism remains: v5's cost 8/12/16 is chosen on the same seven outer-fold
scores being reported; Route A discloses ~20 threshold variants per feature family on its seven
folds. Both need frozen confirmation/nested selection. Small samples and prior match reuse must
remain explicit; a statistical recalculation cannot manufacture fresh matches.

### 4. v5: known pre-kickoff dead frames are not all enforced — confirmed correctness bug

**Location:** `possession/pipeline.py:132–141`.

The structural rule stops at a coarse Smart Tagging kickoff-window start rather than the
organiser-declared kickoff. The regenerated output has **82 non-D frames** in the official
2,934-frame second-half pre-kickoff dead block. Format validation cannot catch this semantic
error. Use the shared official-timeline structural rule, not the approximate tag boundary.
This is a published structural fact, not a manually annotated correction.

### 5. Route A: conflicting owner tags introduce false dead evidence — confirmed logic bug

**Location:** `possession/smart_tagging.py:133–153`;
`possession/test_smart_tagging.py:test_conflicting_claims_cancel_between_teams`.

With both H and A claims, each alive class receives `log(4*p*(1-p))`, while D receives zero.
For informative tags (`p != 0.5`), that term is negative. H/A odds cancel, but alive/dead odds do
not. The existing test only checks H equals A, so it misses the problem.

Synthetic prior `[0.4, 0.4, 0.2]`, agreement 0.8: contradictory phase claims raise dead
probability from **0.20 to 0.28090**, or **0.21845 at the shipped 0.25 weight**. This is unintended
stoppage evidence from ownership disagreement. Neutralize conflicts, or normalize the team-only
contribution to preserve alive/dead evidence; add a three-state regression test. Unioning mirrored
attacking/defending duplicates, which the branch already does, is a good idea but not sufficient.

### 6. Shared input-safety gaps; v5 period boundaries and resampling

Confirmed with synthetic inputs, not malformed real match assets:

- **Both readers leak bad numeric tokens in exceptions.** v5 `tracking.py:105–180` and Route A
  `atd.py:25–51` raise conversion errors containing the input token. Their normal entry paths
  do not consistently sanitize those failures. This contradicts the aggregate-only boundary
  contract. Port main's sanitized parsing/error handling before running either unattended.
- **Duplicate frame IDs silently overwrite** in both readers. v5 also accepts frame ID 0,
  which indexes the last output row. Reject malformed IDs and validate expected frame coverage.
  The byte-identical baseline check establishes correctness on today's valid input, not on these
  failure paths.
- **v5 features cross period boundaries.** `features.py:141–217,229–265,277–335` computes rolling
  windows, ball processing and carry memory over the whole array. In a two-half synthetic fixture,
  changing only the first half alters 12 second-half feature columns and all 500 second-half
  `t_carry_20` values. DFL halves are concatenated without a halftime gap, making this relevant to
  external evaluation even though the target video has a long interval. Reset computations per
  block. Route A's run-aware feature utilities are a useful implementation reference. Main's
  existing feature helpers also deserve this invariant test; this is not a claim they all pass it.
- **v5 SkillCorner interpolation discards a valid exact sample** when its next neighbour is NaN:
  `_upsample()` at `skillcorner.py:28–34` evaluates `0 * NaN`. A synthetic finite sample loses both
  coordinates. Preserve exact observations and avoid bridging period boundaries; then invalidate
  old caches and rerun affected benchmarks.

## Why the external results need caution even without code bugs

**The simulation is part of the hypothesis.** v5 `external.degrade()` erases the ball throughout
true D runs, then synthesizes a stationary restart ball and a short outgoing trail. It does this
on DFL too, replacing its supplied dead-ball trajectory. Live and dead missingness are generated
by different label-conditioned mechanisms. That is a legitimate simulator, not direct access to
truth at inference, but the model can learn simulator-specific clues. The current branch labels
about 95% of target long-gap frames dead (branch-reported); our fixed mask checks show why assuming
that missingness means dead is risky. Matching baseline 0.246 to published 0.2433 is a useful
calibration check, **not proof the simulator is equally realistic for another algorithm**.

**Route A's conditional missingness ratio is not preserved automatically.** The renewal formula
in `train.drop_ball()` uses state-dependent starting hazards, but gaps cross state boundaries.
On a declared synthetic alternating 30 s alive / 30 s dead sequence, ten seeds yield missingness
**47.83% rather than configured 40.7%**, and a D/alive missingness ratio **1.70 rather than 5**.
This does not establish the realised rates on DFL; it shows those rates must be measured rather
than inferred from parameters. Also vary position jitter, correlated drift, player dropout and
slot fragmentation, not only ball gaps. Identity-free feature definitions are not a guarantee of
tracker invariance, especially when velocities still compare the same slot over time.

**SkillCorner is corroboration, not exact independent gold.** v5 `skillcorner._labels()` locates
stoppage/restart boundaries using the same tracked ball's out-of-bounds and slow-motion cues that
feed the predictor. Its possession fields are provider-generated proxies as well. Agreement can
reward imitation of the label-construction heuristic; report it separately from corrected
tracking truth. Preserve native Metrica gaps and raw DFL labels as separate judges.

**Apply standards to main too.** Main's historical external results reuse matches and some fitted
shape coefficients; D007 already documents this. D011/W2 also records using public granularity to
resolve externally flat parameter choices. Latest upstream bans public tuning, including such
tie-breaking: audit/freeze those choices externally rather than treating main as an unquestionable
reference. Latest upstream now permits target self-/unsupervised learning without human labels,
whereas D005 is more restrictive; that policy discrepancy is noted, not silently changed here.

## The reasoning worth keeping

| Priority | Insight | Assessment and smallest useful experiment |
|---|---|---|
| 1 | **v5: speed estimates must survive tracker changes** | Strongest new contribution. One-second displacement, speed relative to the half, slow/fast-player fractions, and separate past/future windows address a real measurement problem. Main uses provider player-speed channels and an absolute slow threshold. Test a frozen dead-evidence replacement while keeping main's ownership and decoder fixed; use relative **and** absolute safeguards because a quiet half can change the normalization. |
| 2 | **v5: vary the tracker model, not just the random seed** | Add smooth versus jittery versus drifting positions, dropout and slot swaps to the shared external judge. Report worst-case and per-match changes, including alive long gaps and tracked-ball stoppages. Preserve real native gaps; do not make the simulator the only judge. |
| 3 | **Route A: modular paired feature ablations** | `lomo(feature_fn, train_fn)` correctly excludes the held-out table from fitting and makes one-component comparisons easy. Ball-free defensive-line, oriented-centroid and opponent-half features extend main's existing shape clues. Test those increments alone before changing classifier and decoder simultaneously. The reported tactical-only gain is more informative than the final combined score, but still needs frozen confirmation. |
| 4 | **Both: contact evidence beats nearest-player flicker** | Carry memory, relative ball/player velocity, reception and kick cues are physically motivated. Much of this already exists in main's last-touch memory and rejected V3 contact experiments. Borrow precise hypotheses, not a claim that another touch model is automatically new or better. Protect genuine short possessions; do not hard-smooth the evaluation truth. |
| 5 | **Route A: tags as soft evidence, not labels** | Deduplication of attacking/defending twins and excluding weak zone codes are useful. Remove public calibration and conflict-induced D first. Main's V4 phase-fusion experiments already failed its external gain gate; a new attempt needs materially better timing/observation evidence, not another target-window weight search. |
| 6 | **v5: separate dead segmentation from team segmentation** | This prevents using D as a cheap route between H and A. Worth a bounded decoder ablation, but an erroneous first-stage D boundary becomes irreversible. Main already limits this issue with its transition costs and team/dead evidence design; no reason to replace both layers at once. |

**Proposed next step, not an adopted model:** predeclare at most two tracker-robust dead-evidence
recipes from v5. Fit every learned component excluding the outer match; retain raw external
labels, common adapters and shared clean/noisy cases; reserve confirmation before measuring.
Compare against frozen main and report match-level uncertainty. Explicitly test slow live
build-ups, ball-seen stoppages and long live occlusions. Only an externally selected finalist gets
the unchanged public/stress/reproducibility gate. Do not average the two full submissions or tune
a hybrid on the public dead-ball recall table. Earlier V2 player-only dead models and W2 slow-ball
rules failed transfer: improved observation robustness is the new hypothesis that needs evidence.

## Verification commands and retained artifacts

Source snapshots and review scripts are retained locally under ignored
`.orchestration/branch-review/`; frame-level artifacts are under `MTAH_WORK_DIR`, never there.
No source snapshot contains competition assets. The project interpreter ran both isolated branch
suites with each snapshot as cwd/PYTHONPATH: `-m unittest discover -s possession -p 'test_*.py'`.

```sh
# Aggregate-only verification; scripts require the retained source snapshots.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 scripts/py .orchestration/branch-review/check_inference.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 scripts/py .orchestration/branch-review/check_stress.py
scripts/py .orchestration/branch-review/probe_v5.py
scripts/py .orchestration/branch-review/probe_route_a.py
scripts/py -m unittest discover -s possession -p 'test_*.py'
python3 -m unittest discover -s scripts -p 'test_*.py'
```

The CLI wrapper captures/discards raw subprocess logs and returns only aggregate JSON, including
on failures. The probes use synthetic inputs or the upstream public labels. Branch code and all
production code/configuration remain unchanged; this review changes documentation only.
