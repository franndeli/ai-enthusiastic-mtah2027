# V4 — automatic phase ownership evidence

**Outcome: both recipes rejected at external development. No V4 target scoring, confirmation,
parameter rescue, or production change.** The current public spellF1 remains 0.6000.

## Why this experiment

The supplied automatic Smart Tagging includes attacking/defending phases, whereas v3c mainly uses
set-piece information. A coarse phase ownership prior could supplement tracking during gaps or
uncertain control. These tags are derived from the same original video/tracking, not independent
Bayesian evidence and not ground-truth annotations.

Two supervised source-only workers implemented a pure phase model and an external event adapter.
Coordinator reviewed them, implemented evaluation/integration checks, and ran actual work on
RunPod. No competition assets left the Mac, no model was fitted on the target, and no labels were
manually added/corrected. No new dependencies.

## Two fixed recipes

Both use publicly documented attack codes (build up, progression, attacking transition, creating
chances, long ball) as soft evidence for the tagged team; explicit defending phases as evidence
for the opponent. Unknown teams, spatial/count-only codes and set pieces do not vote. Intervals
are half-clipped, with a fixed two-second edge taper. Duplicate code/team intervals are max-pooled;
conflicting families are averaged, not counted as independent observations.

Add at most one nat to the baseline H/A logit, retaining v3c's total team confidence cap, dead
evidence and decoder. The only difference is the gate:

- `phase_gap`: centered one-second missing-ball fraction, computed within each half.
- `phase_uncertain`: `clip(1 - abs(control_balance)/0.6, 0, 1)`.

The coordinator retained the total confidence cap and fixed duplicate-vs-conflict weighting during
source review, **before any real-match score**. No recipe changed after scoring.

## Genuine external observations, with a timing caveat

Constructed coarse team intervals from actual public IDSSE pass/cross/shot/claim events, never
from evaluated possession labels. Consecutive same-team observations no more than eight seconds
apart merge; endpoints expand two seconds. Home/away use explicit metadata roles.

Timestamp conversion uses the first ball frame's N/T per half, verifies a 25 Hz clock sample and
exact agreement with cached half boundaries. All seven feeds passed structural checks: 772–1,034
supported observations and 235–275 coarse intervals per match, no unknown-team observations.

The pass-speed lag diagnostic raised a real warning: five matches have peaks approximately 2–3 s
after the raw event time. Coordinator investigated before scoring rather than selecting a lag:

- All 180 supplied CalculatedFrame/CalculatedTimestamp pairs independently check the clock mapping;
  179 residuals are numerical precision and one is one frame.
- Kloppy's loader retains only ball-bearing frames, consistent with the cache's frame origin.
- The [dataset paper](https://doi.org/10.1038/s41597-025-04505-y), especially *Synchronicity between
  event and position data*, explicitly distinguishes timestamp coordinate mapping from inaccurate
  manual event annotations. Sharing a clock does not make EventTime an exact contact timestamp.

Thus the input coordinate system is established, but the observations are temporally noisy. The
experiment retained original timestamps without a searched offset or ownership-label alignment.
It tests a coarse noisy observer, not a precisely synchronized touch detector. Manual events are
not assumed uniformly cleaner than target automatic tags. The additional degradation below is a
simulation, not a measured target error process.

## Paired development design and results

Same four development matches and leave-one-match-out shape refitting as V3; evaluated match
excluded from every fit. Phase models fit nothing. Clean tracking uses original observations.
Noisy conditions combine tracking seeds 11/29 **and** independently seeded endpoint jitter ±2 s,
30% interval dropout, and 10% whole-interval team flips. No undegraded-noisy rescue condition.
Matches, not seeds, are the aggregation unit. Reused-match intervals remain descriptive.

| Recipe | Clean spellF1 | Clean gain | Noisy spellF1 | Noisy gain | Noisy macro gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Refitted baseline | 0.38927 | — | 0.30732 | — | — |
| phase_gap | 0.38927 | 0.00000 | 0.32288 | +0.01556 | +0.00131 |
| phase_uncertain | 0.40815 | +0.01888 | 0.32797 | +0.02065 | +0.00249 |

Both noisy gains are positive on all four matches. Their descriptive 95% intervals are
[0.01147, 0.02098] and [0.01488, 0.02483]. **Neither meets the predeclared +0.03 screen.** No
recipe was selected, so no reserved-three confirmation or target check was run. Do not infer that
Smart Tagging has no value: these two fixed fusions failed this particular noisy external proxy.
Nor may their modest gains be presented as an improvement over the target's 0.6000.

SkillCorner's phase labels were not reused as both model inputs and evaluation truth. That would
be a misleading circular benchmark. A tracking-only target stress mask would also leave the
original automatic tags intact, potentially retaining information about masked tracking; this
caveat was declared before scoring, not discovered from a favorable target result.

## Verification and reproducibility

19 phase tests + 23 adapter tests + 12 coordinator tests pass. They include unknown tags, symmetry,
half boundaries, duplicate/conflicting intervals, unchanged D evidence, explicit source exclusions,
paired observation noise, no local real-match evaluation, source hashes, hook restoration, and
first-half/full inference equivalence. The first combined test invocation exposed a scratch-module
name collision; corrected before scoring, and the complete coordinator suite passes.

An initial clock-diagnostic job stopped at its remote-only guard because SSH did not inherit
RUNPOD_POD_ID; the retry explicitly supplied the verified pod ID on the remote command. It did not
run locally or circumvent a data guard.

Ignored scratch: `.orchestration/v4/` (protocol, source, tests, aggregate development report and
negative selection record); managed remote logs: `.orchestration/v3/10-...` through `15-...`.
No V4 fitted model artifact or submission/prediction file was created. Useful commands (actual data jobs on
RunPod only):

```sh
scripts/py .orchestration/v4/events/prepare_cli.py fetch
scripts/py .orchestration/v4/events/prepare_cli.py anchors
scripts/py .orchestration/v4/events/prepare_cli.py audit
scripts/py .orchestration/v4/clock_check.py
scripts/py .orchestration/v4/sync_metadata.py
scripts/py .orchestration/v4/evaluate.py --mode development --out .orchestration/v4/development.json
```

The scratch evaluator requires the verified RUNPOD_POD_ID environment. These are a record of this
checkout's experiments, not a claim that ignored scratch files ship in a fresh clone. Production
reproduction remains the documented v3c command, unchanged.
