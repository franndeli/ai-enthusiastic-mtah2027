# V2 — pushing beyond local rules, 2026-10-22

Status: done — five learned recipes tested, one independently confirmed, none promoted.
Production v3c is unchanged. This is not a claim that v3c is optimal.

## What changed in the investigation

V1 tested seven local rule/decoder variations. V2 instead implemented two externally trained
factorized learners and one fixed composition, with **fitting separated from evaluation for
every learned component**. Three OpenAI models again worked under coordinator supervision:
GPT-6-Astra on stoppages, GPT-6-Sol on ownership, GPT-6.1-Sol on validation and independent-data
research. The latter was reused for a bounded new-provider adapter. Run `run_8d721e3b9383`.
No new dependency, target fitting, manual annotation or production edit.

The shared scratch evaluator refits baseline shape weights only on outer training matches,
passes the same IDs to candidate fitting, and uses identical noisy inputs for both arms.
Development is four-fold leave-one-match-out over J03WMX/J03WN1/J03WOH/J03WOY, clean plus
seeds 11/29. Confirmation uses J03WPY/J03WQQ/J03WR9, clean plus seeds 101/307/911, with
all fitting restricted to the four development matches. These matches are historically reused,
not a pristine dataset. Seeds are averaged within match, then matches weighted equally.
Independent synthetic audit prompted additional hook-restoration and exact paired-grid checks.

## Five declared recipes

- **Player-only stoppages:** shallow regularized HGB over pooled player-speed distributions and
  half-local multiscale temporal activity, no ball features or IDSSE frozen-ball shortcut.
  Two fixed recipes: learned activity replacement and 50% rule-activity blend.
- **Domain-adapted ownership:** merge external H/A runs shorter than 2 s into the preceding team
  within uninterrupted alive blocks, never across D or a half boundary. Train a team-only HGB
  with noisy seeds 1/2, team-swap augmentation/symmetric prediction, fixed temperature 1.
  Two recipes: learned ownership replacement and a fixed 50% rule-team blend. Both retain
  the rule dead evidence and six-nat decoder; final D predictions can still change through
  joint decoding. No fold temperature or evaluation-label fitting.
- **Fixed composition:** combine both 50% blends. Declared after the pure stoppage result but
  before either blend's results were reviewed by the coordinator; no blend grid. Counted as
  an additional development attempt, not independent confirmation.

| Recipe | Clean spellF1 change | Noisy spellF1 change | Development decision |
| --- | ---: | ---: | --- |
| Player-only stoppages | -0.00954 | +0.01025 | Reject: insufficient ranked gain; a clean-match loss exceeds 0.02 |
| Half-blended stoppages | -0.02224 | -0.00321 | Reject: regressions |
| **Coarsened learned ownership** | **+0.06268** | **+0.07456** | Pass numerical screen; selected before confirmation |
| Half-blended ownership | +0.04509 | +0.05227 | Pass screen, not selected |
| Both half blends | +0.01439 | +0.04916 | Pass screen, not selected |

The pure stoppage learner improved noisy dead F1 by +0.04194, but that did not translate into
sufficient possession-spell improvement. It was not promoted using a secondary metric.
All four development matches improved in clean/noisy spellF1 for both ownership recipes.
The coordinator reproduced the selected ownership result with the hardened evaluator.

## Frozen ownership recipe: reserved-match confirmation

The selected recipe was not changed after development or confirmation.

| Metric | Development-refitted rules | Ownership learner | Change |
| --- | ---: | ---: | ---: |
| Clean spellF1 | 0.44747 | 0.48345 | +0.03598 |
| Noisy spellF1 | 0.38329 | 0.43101 | +0.04772 |
| Clean macroF1 | 0.82195 | 0.82882 | +0.00687 |
| Noisy macroF1 | 0.79228 | 0.80083 | +0.00854 |

| Reserved match | Clean spellF1 change | Mean noisy spellF1 change |
| --- | ---: | ---: |
| J03WPY | +0.02554 | +0.02868 |
| J03WQQ | +0.04930 | +0.05698 |
| J03WR9 | +0.03311 | +0.05750 |

This is a substantial reproducible gain on the external task, with no evaluated match in fitting.
It is **not** proof of new-match population performance or target transfer. The three-match
percentile bootstrap is only descriptive: [0.02868, 0.05750]; the exact two-sided sign-flip
p is 0.25. Neither more seeds nor a favorable bootstrap removes the small-sample/reuse limitation.

## Target-domain check: reject promotion

Only this selected recipe was checked on the target, once as a decision experiment. It remained
trained on the four development matches. No runner-up was tested on target to rescue the failure.

| Target diagnostic | Exact production v3c | Frozen ownership | Change |
| --- | ---: | ---: | ---: |
| Public spellF1 | 0.60000 | 0.55484 | -0.04516 |
| Public macroF1 | 0.82715 | 0.80783 | -0.01932 |
| Ten-seed transplant spellF1 mean | 0.55919 | 0.51884 | -0.04035 |
| Transition-centred 5 s spellF1 | 0.40000 | 0.34667 | -0.05333 |
| Transition-centred 10 s spellF1 | 0.41538 | 0.36923 | -0.04615 |
| Transition-centred 20 s spellF1 | 0.47692 | 0.41481 | -0.06211 |

Public predicted possessions rose 11→14 (17 true); full-match predicted possessions 264→335.
Those counts do not establish full-match accuracy. Candidate public 90% interval is
[0.36129, 0.73548]; overlap with v3c does not excuse failing the predeclared non-regression gate.
The result is evidence of a transfer problem, not proof of its precise cause or statistical
proof that the candidate is worse on the private match. No settings were revised using target
outcomes. Candidate format validation passed and repeated inference produced byte-identical CSVs
(SHA-256 `19dbe25830a6c190689a52a9f3519d9f80d3c5d6c2a74bc3cf5c3777b0571e9d`); those engineering
checks do not make a rejected candidate acceptable.

## New-provider evidence

The independent search verified twenty SkillCorner A-League 2024/25 tracking bundles under MIT,
at commit `4340d274572876239c154c90bc507a9b3250a656`, totaling 1,823,448,331 tracking bytes.
Its continuous possession and phase information is algorithmic/provider-derived, **not verified
independently corrected H/A/D truth**. In particular, the phase documentation uses an absence-of-
events heuristic longer than 30 s for some out-of-play decisions; null owner cannot simply be
called true D. It is useful for weak cross-provider corroboration, not a replacement for gold.

Sources: [pinned repository](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656),
[phase specification](https://26560301.fs1.hubspotusercontent-eu1.net/hubfs/26560301/Guides/Phases%20of%20Play/20250216%20-%20Phases%20of%20Play%20CSV%20Specifications.pdf).
The bounded acquisition reserves the first three sorted IDs, 1874553/1886347/1899585 (~266 MB),
with pinned hashes, synthetic-tested 10→25 Hz conversion and no fitting on the new provider.
All nine authorized assets were downloaded sequentially and hash/size-verified: **266,007,186
bytes**. The coordinator reviewed the adapter and independently reproduced its sole comparison.

| SkillCorner match | Weak-label rules spellF1 | Ownership spellF1 | Change |
| --- | ---: | ---: | ---: |
| 1874553 | 0.50448 | 0.50381 | -0.00067 |
| 1886347 | 0.48464 | 0.51138 | +0.02674 |
| 1899585 | 0.44923 | 0.45814 | +0.00891 |
| Equal-match mean | 0.47945 | 0.49111 | +0.01166 |

Mean macroF1 rose 0.61747→0.61930 (+0.00183). This modest, mixed result does not meet the
+0.03 improvement threshold and cannot override the target regression. Native recognized-owner
strict agreement (unknown owner ungraded, predicted D penalized) rose 78.485%→78.806% on average,
but those native groups are another correlated provider output, not a second gold source.

Coverage audit: no missing source frame cells/clocks, timebase mismatches, unknown phase teams
or overlapping phases. One leading phase per match clipped to observed half support as
predeclared. The parser preserves zero/missing coordinates, interpolates only consecutive
same-half observations, recomputes speed at 25 Hz and isolates both feature computation and
decoding by half. Consequently its label-free shape normalization is per-half rather than the
full-match normalization used elsewhere; both arms share that behavior. Half-open phase endpoint
semantics are an explicit convention, not independently verified provider truth. Broadcast
extrapolation, sample selection and weak D semantics limit interpretation. No labels, model
settings or match selection were changed after scores.

## Reproduction and verification

Experimental source and aggregate reports remain ignored under `.orchestration/v2/`; local
frame-level assets/caches remain outside the repository. No submission pipeline imports these
experiments. The following commands refer to this checkout's retained scratch, not shipped code:

```sh
scripts/py .orchestration/v2/common.py --module .orchestration/v2/ownership/candidate.py --recipe ownership --out .orchestration/v2/coordinator_ownership_development.json
scripts/py .orchestration/v2/common.py --module .orchestration/v2/ownership/candidate.py --recipe ownership --mode confirmation --out .orchestration/v2/coordinator_ownership_confirmation.json
scripts/py .orchestration/v2/common.py --module .orchestration/v2/combined/candidate.py --recipe factorized_half --out .orchestration/v2/combined/aggregates.json
scripts/py .orchestration/v1/validate.py target --plugin .orchestration/v2/ownership_target.py
scripts/py .orchestration/v2/skillcorner/adapter.py fetch
scripts/py .orchestration/v2/skillcorner/adapter.py audit
scripts/py .orchestration/v2/skillcorner/adapter.py evaluate
scripts/py .orchestration/v2/check_production.py
```

Coordinator checks: selected development and confirmation, pure stoppage, combined and
SkillCorner evaluations exited 0 and reproduced the reported results. Synthetic suites passed:
common 10, extra ownership-boundary 2, independent audit 11, stoppage 8, ownership 4,
combination 2, SkillCorner adapter 13; package 94 and guard 12. **156 tests total.**
Official validator exit 0 for both the rejected target artifact and unchanged production.
Production cold-cache/warm-cache regeneration matched the original SHA-256
`374f198f07ccecbc5f7b6231235c40eb6bbda3c66c391687bff6de5dc22d7758`.
Production diff empty; no commit or push.

All four dispatches settled. Ownership and the reused validation/acquisition worker were
released; Orca preserved the settled stoppage terminal as `user_owned` / `user_takeover`, so the
coordinator did not close it. No coordinator-owned worker remains pending.

## Outcome and next scientifically useful step

The first review did not exhaust the available approaches: this wave found a substantial,
repeatable external-task improvement and tested it on a new provider. It also located the
remaining failure more precisely: transfer from IDSSE to the competition domain, not merely
insufficient decoder tuning. Exact causes remain unproven. Repeated target-window tuning would
now contaminate the evidence, so these frozen recipes are rejected rather than adjusted there.
The acquired adapter supports future multi-provider training/source-held-out experiments, but
new weak labels must stay distinguished from gold and the remaining untested matches must be
reserved before model development. No claim of a better deployed model is made.
