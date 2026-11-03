# BR2 — tracker-robust dead evidence

Status: complete — both fixed recipes rejected at external development; no production promotion.
Production remains main v4 (`60d0bfc`).
User priority: generalization, never an apparent win obtained by fitting the public five minutes.

## Hypothesis and scope

BR1 identified a credible measurement problem: short-span speed estimates change with a tracker's
jitter/smoothing. V2's earlier player-only HGB used supplied speeds, trained on IDSSE alone, and
failed transfer. This experiment instead recomputes half-local one-second displacements, uses
both absolute and relative activity, and trains across provider/noise conditions. No ball-dependent
learned feature, synthesized dead-ball trajectories, target calibration or label coarsening.

Exactly two recipes, no hyperparameter search:

- `replace`: replace only v4's bias + slow-player activity contribution with an externally fitted
  logistic activity head. Fit with the unchanged rule-ball evidence as a fixed logit offset.
- `half`: 50/50 blend of learned and original activity logits; same fitted head, no blend search.

Keep team logit, ball/tag clues, confidence caps and Viterbi settings identical between arms.
The head sees 25 features: median speed, median speed / half median (denominator floor 0.25 m/s),
fractions <1.5 and >4 m/s, observed-speed count /22; each averaged over centred +/-1/4/15 s and
strictly previous/next 3 s. Displacements use +/-0.5 s (rounded to frames), reject >12 m/s and
missing intermediate observations. All calculations reset per half; missing observations are
imputed with training means. The model has a missingness indicator for each of these 25 columns.
Normalise with training-only weighted mean/std; logistic intercept unpenalized, L2=0.001 on
standardized coefficients, weighted mean loss, maximum 500 L-BFGS iterations, fixed initialization.
Use every fifth graded frame, equal weight per provider, match and training condition. No class
rebalancing or post-fit temperature/calibration.

## Fixed external protocol

- Development: IDSSE `J03WMX,J03WN1,J03WOH,J03WOY`, Metrica `Sample_Game_1,Sample_Game_2`.
- Reserved confirmation: IDSSE `J03WPY,J03WQQ,J03WR9`, Metrica `Sample_Game_3`.
- Development: leave one whole match out, train on the other five. Refit *all* learned components,
  including the comparator's four shape weights, without the evaluated match. The same fitted
  ownership component is used by both arms. No evaluation labels enter inputs or inference.
- Training conditions: clean plus the four scenarios below, seeds 0/1. Development scoring uses
  clean plus seeds 11/29; confirmation uses 101/307/911. Same corrupted observations for both arms.
- Keep IDSSE raw H/A/D; keep main's Metrica adapter unchanged. Metrica labels are event-derived
  approximations, not exact corrected-tracking gold. Preserve native ball gaps in all conditions.
- `standard`: existing `external.inject_noise`, with 41% additional ball dropout on IDSSE and
  15% on Metrica, ball jitter 0.3 m and slot fragmentation/duplication; native player speeds retained.
- `smooth`, `jitter`, `drift`: start from the same standard noisy copy; add player XY noise of
  0.03 m independent / 0.30 m independent / 0.30 m with 1 s correlation. Add unconditional player
  gaps (7% expected occupancy, mean 3 s), recompute supplied speeds with a half-local two-frame
  central difference. Noise never uses labels, never crosses half boundaries, never restores gaps.
  Measure realized missingness and speed summaries rather than assuming intended rates are met.

## Selection, confirmation, and stopping

Primary comparison is **standard-noise spellF1**, averaging seeds within match first, then matches.
A recipe must meet all of: mean gain >=0.03; paired match-bootstrap 95% lower bound >0; positive
mean gain on each provider; mean clean loss <=0.01; each match clean/standard-mean loss <=0.02;
mean macroF1 loss <=0.01 for clean and standard. Tracker-shift scenarios additionally require
pooled noisy gain >=0.03 and no scenario mean loss >0.01. Report all match/scenario deltas,
match-bootstrap interval and exact sign-flip p; these are descriptive on historically reused data.
No treating noise seeds or possessions as independent matches.

If both pass, select the higher standard gain, ties to `half`; otherwise select the sole passer.
If neither passes, stop: no confirmation, target evaluation, threshold rescue or third recipe.
Freeze that finalist before confirmation. Fit on the six development matches and evaluate the
four reserved matches once with the same numerical gate. Also fit excluding an entire provider:
IDSSE-development -> Metrica-development and reverse. Require positive standard gain and no clean
mean loss >0.01 in each direction. Failure ends this attempt; no fallback candidate at confirmation.
The reserved matches and both providers have historical exposure across main/colleague work;
these are recipe-held-out checks, not pristine/new-provider proof. Do not claim guaranteed private
performance. Meaningful uncertainty or inconsistent transfer is a reason to retain v4.

Only a passing external finalist may be fitted on all ten external matches and evaluated locally
against frozen production with the unchanged SPEC learned-component target/stress/reproducibility
gate. No public fitting, tuning, count matching or tie-breaking. Preserve existing submission files.

## Execution and safeguards

Single coordinator. Pure module `possession/robust_dead.py`, external-only evaluator
`scripts/robust_dead_eval.py`, synthetic tests; neither imported by production `run.py`.
All intermediate features/predictions/models remain outside Git. Aggregate results only in docs.
Local work: code, synthetic tests, lightweight inference. Heavy evaluation: one visible RunPod
job, at most 3 hours / $3, delete on exit/failure and verify the exact owned resource is absent.
Only allowlisted public IDSSE/Metrica caches and source code may be uploaded. No competition
assets, target caches, public target labels or submissions are transferred. No dependency additions.

## Implementation checkpoint (before external results)

- Added the pure experimental module and evaluator; no production imports or configuration edits.
- 23 new synthetic tests cover half isolation, missing intermediate observations, identity jumps,
  team/ball independence, units, fit determinism, preserved ownership odds, fit/evaluation exclusion,
  paired conditions, seed replication, provider regression gates and the fixed tie-break. Two
  audit tests added during remote execution establish that changing evaluation labels cannot
  change predictions, and a development rejection never touches reserved matches or the target;
  no experiment source/configuration changed.
- Local package suite: 131 tests pass; guards: 12 pass. One initial test incorrectly required
  past/future windows to be finite at half boundaries; corrected the assertion to interior frames
  (boundary missingness is intentional), then the full suite passed.
- Source/config/adapter/scorer fingerprint before evaluation:
  `1fab2ea6d4597fde2722601b7b1943f36384c54b411ad5baabcca1d32719427b`.
- Ten required public external caches present. No target input or target output is part of the
  allowlisted transfer. No new dependencies.
- Managed task `3d878181-0f57-4ffc-9db4-3ce8cb2dc6d3`, visible terminal
  `term_5749733c-06b8-4840-91c2-4d6cb1ea2627`. Runner:
  `scripts/py .orchestration/br2/remote.py`; remote command:
  `scripts/py scripts/robust_dead_eval.py`. Both numerical libraries bounded to two threads remotely.

### Infrastructure retry (no model results)

First task exited 1 with `StopIteration` before provisioning: the old RTX 2000 Ada SKU is no
longer in the RunPod catalog. Independent pod inventory returned zero BR2-owned resources.
The launcher now chooses the cheapest available secure instance with quoted rate <=$0.60/h
(current catalog: RTX PRO 4000 Blackwell, $0.57/h), retaining the actual-rate safety cap $0.75/h,
$3 total bound and 3 h limit. No model/evaluator/configuration change or candidate selection.
Fingerprint rechecked unchanged. Retry managed task `db431222-1f11-44eb-ad40-dcc5dc33f078`,
visible terminal `term_619b5249-7151-46b1-a63e-9767e2590a23`. That retry selected an available
RTX 3090 quote ($0.50/h), but provisioning was rejected before any pod was created. Independent
BR2 resource inventory again empty. The initial wrapper discarded API error detail, so the exact
cause of that rejection was not established.

Added safe API diagnostics (predefined categories/status codes only, never raw request/error
payloads), verified with a synthetic credential-echo test. Capacity errors may try at most three
secure choices within the same cap, only after reconciling that no resource was created; other
errors stop. Frozen experiment fingerprint rechecked unchanged. New task
`6e8b95df-91d4-4836-bd26-f105730ba7ca`, visible terminal
`term_2e6f14da-ab4c-49bf-b74d-94e1aff98022`.

That job provisioned pod `63o49s1hyfi591` ($0.57/h), uploaded the ten public caches
(295,864,460 bytes), installed pinned dependencies and passed remote synthetic tests. The evaluator
then failed at its pre-training RunPod guard because SSH did not inherit `RUNPOD_POD_ID`.
No training or scoring ran. Deletion exit 0 and an independent API inventory confirm absence;
lease 0.03117 h, estimated compute $0.01777 (not a reconciled invoice).

Launcher fix only: pass the pod ID obtained from this job's API receipt into the SSH environment,
then check Linux platform, remote cwd and that ID before invoking the unchanged evaluator. Do
not set the marker for local execution. Subtract 113 s from the remaining three-hour runtime and
reserve $0.018 of the $3 budget. Fingerprint remains unchanged. Current task
`8d23d0ec-5655-4770-bd43-e9e8f122d523`, terminal
`term_0f8ad7ab-2902-4446-be13-dd324e2b1e48`.

## Results — no recipe passes the fixed development gate

Successful job: `8d23d0ec-5655-4770-bd43-e9e8f122d523`, exit 0. Six excluded-match folds,
54 paired observation conditions, 108 candidate/condition scores. Training/evaluation seeds and
inputs followed the frozen protocol. No code/configuration fingerprint change. These are comparisons
with **cross-fitted v4**, not the historically pooled-shape external scoreboard.

Mean spellF1 over the six matches, seed-averaged within match:

| Condition | Cross-fitted v4 | Replace | Delta | Half blend | Delta |
|---|---:|---:|---:|---:|---:|
| Clean | 0.51366 | 0.52419 | +0.01053 | 0.52314 | +0.00948 |
| **Standard noise (primary)** | **0.41577** | **0.42990** | **+0.01413** | **0.42767** | **+0.01190** |
| Smooth-position shift | 0.37098 | 0.42349 | +0.05251 | 0.41336 | +0.04238 |
| Independent jitter shift | 0.27169 | 0.40771 | +0.13602 | 0.27204 | +0.00034 |
| Correlated drift shift | 0.28370 | 0.41797 | +0.13426 | 0.29765 | +0.01394 |

- `replace`: primary match-bootstrap 95% interval **[+0.00709, +0.02048]**, two-sided sign-flip
  p=0.03125. Standard-noise gain on IDSSE +0.01622, Metrica +0.00994. All six primary match deltas
  positive; minimum +0.00024. Two clean regressions, worst -0.01249, within the predeclared
  per-match tolerance. Standard macroF1 +0.02529; dead F1 0.81355 -> 0.85484.
- `half`: primary interval **[+0.00996, +0.01361]**, sign-flip p=0.03125. Standard-noise gain
  on IDSSE +0.01232, Metrica +0.01106. All six primary deltas positive; minimum +0.00789.
  Worst clean loss -0.00426. Standard macroF1 +0.02091; dead F1 0.81355 -> 0.84634.
- **Both fail the required primary +0.03 gain.** The half blend also fails the >=0.03 pooled
  noisy gain requirement (four-scenario mean +0.01714; replacement +0.08423). The gate was not
  weakened after seeing the positive but smaller primary effects.
- Selection: **none**. No reserved-match scoring, no provider-excluded confirmation, no all-data
  deployment fit, no candidate target/public/stress check, no fallback or third recipe. Reserved
  caches were transferred for potential confirmation but not loaded by development fitting/scoring.

### What was learned, without overclaiming

The tracker-robust measurement hypothesis has supporting external evidence, particularly for
replacement under speed-channel shifts. Under jitter/drift, baseline dead F1 falls to zero while
replacement gives about 0.85. However, those synthetic conditions intentionally stress short-span
speed estimation; they do not establish the frequency or severity of that problem on the target.
The primary judge must not be replaced by the stress regime on which the candidate looks best.

The half blend is not automatically safer: it retains enough corrupted original activity evidence
to lose most of the jitter/drift benefit. Do not search a different blend on these outcomes or on
the public window. The 1–1.4 percentage-point primary gains are positive evidence on reused
external matches, not a proof of target generalization or grounds to silently relax the threshold.

Measured simulation checks (averages over matches/seeds): IDSSE ball missing 0% clean / 43.15%
noisy, Metrica 40.07% clean / 48.47% noisy; extra scenarios retain identical ball masks. Added
player gaps reduce noisy tracked objects from about 22.45/22.57 to 20.88/20.98. Supplied player
mean speeds rise from about 1.8 m/s to 6.9 m/s under jitter and 2.7 m/s under drift, illustrating
why absolute-speed dead rules fail in those constructed regimes. These are diagnostics, not new
target-domain facts or a new calibration objective.

### Independent closeout verification

- Recomputed every summary and selection locally from the **aggregate-only** fold records:
  exact equality with remote results; all six fit/evaluation exclusions and all 108 expected
  candidate cases verified; selected recipe remains null. No heavy local training/evaluation.
- Final tests: `scripts/py -m unittest discover -s possession -p 'test_*.py'` -> **131 pass**;
  `python3 -m unittest discover -s scripts -p 'test_*.py'` -> **12 pass**;
  `git diff --check` passes. Core production files/config have no diff.
- An optional inline cold/warm production-regeneration command was blocked by the data guard
  for naming the private-root environment variable. Stopped that check without alternate-path
  retries. Therefore **no new cold/warm reproduction claim** is made in BR2; BR1's earlier hash
  remains historical evidence. No candidate output was generated locally.
- Successful pod `whgc4rac0v7uqf`: deletion exit 0, independently absent, zero remaining BR2
  resources. Lease 0.15747 h, estimated compute **$0.07873**; including the earlier provisioned
  failure, rounded cumulative compute estimate **$0.09673**. These are runtime/rate estimates,
  not reconciled invoice/storage totals. Both provisioned pods are confirmed deleted.
- Experimental code/tests retained, deliberately not wired into `run.py`; all model/feature
  training artifacts were remote and disappeared with the pod. Aggregate logs remain in ignored
  `.orchestration/br2/aggregates.json`. No commit or push.

Next research direction, not an accepted model: a separate predeclared ablation of Route A's
additional formation features under the same exclusion/confirmation discipline. This is not a
rescue of BR2 or permission to tune the public five minutes.
