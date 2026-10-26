# OpenAI-only generalization review — 2026-10-22

## Decision

**Retain v3c. No production model, parameters, feature code or decoder changed.** Seven
predeclared alternatives were implemented in ignored scratch space and rejected. The strongest
observed noisy-development spellF1 gain was **+0.01635**, below the predeclared +0.03 gate.
This does not prove the alternatives overfit; it means they did not establish a sufficiently
large, validated generalization improvement. Nor does it prove v3c is the best possible model.

Three supervised Codex agents used distinct OpenAI models, confirmed by Orca launch receipts:
GPT-6.1-Sol (independent audit), GPT-6-Astra (control/evidence), GPT-6-Sol (decoder), all high
reasoning effort. Run `run_5871893e60b7`; all three completed and were released. Coordinator
reviewed experiment code, reran both development evaluations and all synthetic checks, and owned
the final rejection decision. Match content and frame-level outputs remained local; only code,
public documentation and aggregate metrics entered agent contexts. No dependency was added.

## Frozen experiment design

Before scoring: at most four hypotheses per experimental worker, no parameter grid, no competition
window used for candidate selection. Development: first four sorted IDSSE IDs (J03WMX, J03WN1,
J03WOH, J03WOY), clean tracking and paired noise seeds 11/29. Equal weight per match; average seeds
within match. Production config stayed frozen. These scores are **exploratory**: its shape
coefficients were historically fitted using all seven external matches.

Confirmation was reserved for J03WPY/J03WQQ/J03WR9 with seeds 101/307/911 and shape coefficients
refitted using only the four development matches, identically for both comparison arms. No
alternative reached confirmation. Coordinator ran the development-refitted comparator alone to
verify this fitting path; those results are not a new production candidate or a claimed gain.
No rejected alternative was scored on the target/public window or target stress masks.

## All attempted alternatives

Mean spellF1, reproduced by the coordinator:

| Alternative | Clean | Noisy | Noisy change | Decision |
| --- | ---: | ---: | ---: | --- |
| Frozen v3c reference | 0.38904 | 0.30701 | — | Keep production |
| Fast confirmed control | 0.41461 | 0.32180 | +0.01479 | Reject: below +0.03 |
| Agreeing gap endpoints preserve ownership | 0.38786 | 0.29948 | -0.00753 | Reject: regression |
| Control evidence takes priority over shape | 0.40593 | 0.31458 | +0.00757 | Reject: below +0.03 |
| Fast control + control priority | 0.41874 | 0.32336 | +0.01635 | Reject: below +0.03 |
| Destination-supported switch cost | 0.38797 | 0.30300 | -0.00401 | Reject: regression |
| Nine-frame median of team contrast | 0.38904 | 0.30680 | -0.00021 | Reject: regression |
| Lower team switch cost + 12-frame duration floor | 0.39671 | 0.31180 | +0.00479 | Reject: below +0.03 |

Fast control used a fixed 13-frame window, at least eight exclusive proximity votes, distance
<2 m and opponent margin >0.75 m; ball speed <=8 m/s or unknown. Control priority attenuated
shape by `clip(1-abs(control_balance)/0.3, 0, 1)`. Gap agreement floored memory at one vote
only for equal nonzero controllers separated by <=30 s. Destination-supported switching used
`1.25 - 0.65*clip(destination_margin/log(3), 0, 1)` as a team-cost multiplier. The duration
variant changed only H/A cost to 3.5 nats and required 12 frames per state; D costs stayed six.
No settings were revised after their scores were observed.

Best rule candidate clean/noisy macroF1: 0.82530/0.77886 versus 0.82228/0.77716 reference.
Best decoder candidate: 0.82230/0.77744. Small frame-level gains do not override failed spellF1
acceptance. Cross-dataset label granularity and incomplete simulated noise remain limitations.

## Validation audit: historical claims need qualification

1. **Pooled shape-fit overlap.** `scripts/idsse_eval.py` evaluates fixed production coefficients;
   `scripts/fit_shape_weights.py` fits those on all seven evaluated matches. It is not
   leave-one-match-out evaluation for this learned component.
2. **Calibration leakage in the rejected learned experiment.**
   `possession/learned.py::fit_with_calibration` fits classifiers excluding their evaluation match,
   but selects one temperature using labels from every fold and assigns it back to those folds.
   A synthetic test changed only an evaluation fold's labels and changed its temperature.
   This experimental model is not in the production path. Future evaluation must calibrate
   inside outer training folds, or freeze temperature before evaluation.
3. **Insufficient independent evidence.** All seven IDSSE matches were historically reused;
   three reserved matches are not a pristine holdout. Three paired differences cannot supply
   distribution-free two-sided 95% evidence: minimum exact sign-flip p is 0.25. A positive
   percentile bootstrap interval on three positive values would not solve that problem.
4. **Public uncertainty is not new-match uncertainty.** Official 5th/95th percentile intervals
   are 90%, not 95%; masks and seeds reuse the same 17 target possessions. They are useful
   diagnostics, not additional independent matches or proof of no overfitting.

Historical scores remain recorded, but `docs/TECHNICAL.md` now qualifies their interpretation.
The next credible promotion requires leakage-free fitting/calibration and more untouched,
comparable external-match evidence—not repeated tuning against the public window. Unimplemented
validation repairs and broader external-loader robustness findings remain future work.

## Independently reproduced production baseline

- Public spellF1 **0.6000**, official 90% interval **[0.3857, 0.7823]**; macroF1 **0.82715**.
- Public predicted/true possessions: **11/17**. Full-match prediction: **264** possessions,
  mean **14.61 s**; full-match correctness cannot be measured without private ground truth.
- Ten-seed transplant spellF1 mean/min **0.55919/0.42963**.
- Transition-centred 5/10/20 s spellF1: **0.40000 / 0.41538 / 0.47692**.
- Official format validator exit **0**; two regenerations were byte-identical, SHA-256
  `374f198f07ccecbc5f7b6231235c40eb6bbda3c66c391687bff6de5dc22d7758`.

## Verification and retained artifacts

Coordinator commands (scratch is ignored and available in this checkout, not shipped in clones):

```sh
scripts/py .orchestration/v1/decoder/evaluate.py
scripts/py .orchestration/v1/rules/evaluate.py
scripts/py .orchestration/v1/validate.py target
scripts/py .orchestration/v1/validate.py external --baseline-only
scripts/py -m unittest discover -s .orchestration/v1/audit -p 'test_*.py'
scripts/py -m unittest discover -s .orchestration/v1/decoder -p 'test_*.py'
scripts/py -m unittest discover -s .orchestration/v1/rules -p 'test_*.py'
scripts/py -m unittest discover -s possession -p 'test_*.py'
python3 -m unittest discover -s scripts -p 'test_*.py'
```

Results: both evaluations exited 0 (48 decoder and 60 rule comparisons, including repeated
references); audit/decoder/rule synthetic tests **6/4/8 passed**; package tests **94 passed**;
guard tests **12 passed**. Official `evaluation/validate_submission.py` was invoked through a
local wrapper resolving its input via `io.work_dir()` and returning only exit status.
`git diff --exit-code -- possession run.py config scripts evaluation` passed: production unchanged.

Detailed aggregate-only tables, frozen hypotheses, code and worker reports remain under
`.orchestration/v1/`; frame-level baseline verification output is outside the repository.
No commit or push performed. No candidate was accepted, and no improvement or best-model claim
is made.
