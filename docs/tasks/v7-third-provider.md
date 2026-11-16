---
id: task-v7-third-provider
type: task-spec
status: done
title: W7 — third-provider generalization check and retraining (SkillCorner, 20 matches)
parent: goal
references:
  - decisions
  - external-judges
  - possession-pipeline
---

# W7 — does v6 transfer to a provider nothing was built on?

## Request

User (2026-11-09): iterate on the model and make sure it generalizes in the submission; review
every spec, branch and document first.

## Why this wave

Review of the repository state (v6, D015/D016) and of the teammate branch:

- Every v6 choice (rule scalars, dead residual, verifier, the PathCRF weight) was selected on the
  same ten matches of two providers, whose "noisy" rows are our own synthetic noise model. The
  only real third-provider look in W6 was three SkillCorner matches with coarse phase labels, where
  the verifier was slightly negative.
- Nested v6 diagnostic on the five PathCRF-unseen matches (`.orchestration/v7/diag.py`, reproduces
  `scripts/eval_v6_external.py` to four decimals): with the true dead frames v6 gains +0.04..+0.08,
  with the true team +0.21..+0.29. Recall of possessions under 2 s is 0.14-0.23 and of 2-4 s ones
  0.43-0.53; 55-63 % of missed short possessions are swallowed by the other team's spell.
  The remaining loss is short-possession ownership, which is exactly where label semantics and
  tracker noise differ most between providers.
- The teammate branch built a 20-match SkillCorner judge (`docs/research/head-to-head-skillcorner.md`
  there): real broadcast tracking, ball undetected on 40.6 % of frames (target 40.7 %), seen on
  ~30 % of dead and ~90 % of alive frames, 16 players per frame, possessions of 7 s median with
  16 % under 2 s (public window: 2 of 17). The official baseline scores 0.233 there against 0.2433
  on the real match. Main never used 11 of those matches for anything.

## Product

Done means: (1) `possession/skillcorner.py` + `scripts/fetch_skillcorner.py` make the 20 matches a
first-class judge with synthetic tests; (2) a per-component measurement of production v6 on it;
(3) a new production version only if a predeclared candidate passes the bar below, reproducible
and validated; otherwise v6 stays and the measurements are recorded.

## Protocol (declared before any pipeline score on SkillCorner)

Matches sorted by id. **DEV** = the first ten (1874553 … 1996436; nine of them were touched as
weak corroboration in V2/V3/W6, never for selection). **TEST** = the last ten (2006229 … 2017461),
untouched by main; looked at once per frozen finalist, at most two finalists, nothing changed
after a TEST look. Tracking is used natively (no noise injection).

Labels (`possession/skillcorner.py`): primary **event** (team of the last player possession carried
forward, D from game interruptions), secondary **phase** (provider phases; D elsewhere). Both are
provider-derived proxies, not corrected truth; a result must hold on the primary and not reverse
on the secondary.

- **E1 measurement (DEV)**: production v6 stage by stage — v5 config, tuned rules, + fresh exit,
  + dead residual, + PathCRF, + verifier — spellF1 / macroF1 per stage, paired by match.
- **Candidates** (fixed list):
  - C1 `3p`: v6 architecture unchanged; the dead residual and the verifier are fitted with the
    SkillCorner DEV matches as a third provider (event labels, native tracking, equal provider
    weight in the residual; every training row's residual from a model that excludes its match).
  - C2 `drop`: v6 minus every learned component that E1 shows negative on DEV under both label
    kinds (subtraction only, no new parameter).
  - C3 `3p-rules`: C1 with the rule scalars re-selected by `scripts/tune_rules.py` on three
    providers. Only if C1 passes DEV and time allows; never tuned on TEST.
  On DEV, candidates that train on SkillCorner are scored by two-fold cross-fitting (sorted halves).

## Promotion bar (candidate against production v6)

1. SkillCorner TEST, event labels: mean spellF1 gain ≥ +0.01, paired match-bootstrap 90 % interval
   above 0, at least 7 of 10 matches up; phase labels: mean gain ≥ 0; mean macroF1 not lower by
   more than 0.005.
2. Metrica + IDSSE, nested whole-match exclusion on the five PathCRF-unseen matches, clean and
   fresh seeds 2/3: none of the four judge means lower than v6 by more than 0.005.
3. Synthetic tests, official validator, byte-identical cold/warm regeneration. The public window
   is reported with its interval as a correctness check, never a selector (D012, D016).

If nothing passes, v6 stays.

## Unconfirmed assumptions

- The event labels approximate the competition's corrected-tracking owner (same caveat as the
  Metrica and IDSSE labels); stoppage edges are placed with the provider's tracked ball.
- Provider-extrapolated off-camera players behave like the target's tracked players for the
  ball-free components.

## Outcome

v6 retained (D017). C1 failed bar 2, C2 was empty, C3 was not run. Results:
`docs/research/third_provider.md`, `docs/PLAN.md` W7.
