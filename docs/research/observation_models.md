# V3 — physical contact and spatial gap evidence

**Outcome: rejected for deployment; v3c remains unchanged.** This wave found an external-task
improvement, not an improvement over the competition-window 0.6000. No target-driven retuning or
runner-up target scoring followed the failure.

## Design and compute

Two supervised Claude workers delivered contact and temporal candidates, then a source-balanced
variant and a source/mock evaluation audit. Codex's initial launch stopped at an update prompt
before receiving the task; a supervised retry used Claude. Coordinator implemented spatial gap
inference and one fixed composition, integrated the evaluator, and owned all scoring/selection.
Orca run: `run_d89006b3db08`. The earliest launcher attempt was rejected with `consumer_fenced`
at 18:23 UTC because its coordinator was not bound to the run (task `97611595-ee5f-4564-a234-e2b745c21dad`).
This precedes the later supervised launches and completed research; its delayed failure notification
did not require restarting workers or paid compute.

All substantial fitting and external evaluation ran on a bounded RunPod instance (quoted
$0.24/hour, initial $3 / three-hour cap). Only public external data and source code went there;
competition assets stayed on the Mac. Local work consisted of source review, small synthetic tests
and bounded single-thread target inference (native/reproducibility 12.0 s; stress 29.2 s).
No new project dependency, manual annotation, target pseudo-label fitting, private ground truth,
or cloud-LLM access to match content. Existing uncommitted V1/V2 work was preserved.

Development used four-fold cross-fitting on J03WMX/J03WN1/J03WOH/J03WOY, clean observations and
paired noise seeds 11/29. The three other IDSSE matches used only those four for fitting and seeds
101/307/911. Baseline shape coefficients were refit within each permitted training set. Seeds
were averaged within match, then matches weighted equally. All seven IDSSE matches were
historically reused; the reserved three are not a pristine holdout.

## Seven counted recipes

| Recipe | Mechanism | Clean spell gain | Noisy spell gain | Development screen |
| --- | --- | ---: | ---: | --- |
| contact_filter | Velocity-compatible contacts, kick impulses, clipped causal ownership memory | +0.07287 | +0.04079 | Pass |
| deflection_flight | Contact filter plus post-contact deflection gate and flight hold | +0.03966 | +0.01253 | Reject |
| residual_rule | Symmetric temporal HGB, bounded 0.5-nat correction to rule ownership | +0.01379 | +0.02785 | Reject |
| gated_learned | Temporal HGB, four-second external coarsening, visible-contact quality gate | -0.01771 | -0.00389 | Reject |
| source_balanced | Same gated model with equal IDSSE/SkillCorner training weight | -0.01152 | +0.00086 | Reject |
| spatial_gap | Team geometry around last/next/linear ball anchors, learned correction only in gaps | 0.00000 | +0.01869 | Reject |
| **contact_spatial** | Fixed missingness-weighted composition of contact_filter and spatial_gap | **+0.07287** | **+0.06396** | **Selected** |

Reference clean/noisy spellF1: 0.38927/0.30732. The selected composition scored 0.46213/0.37128;
noisy macroF1 improved by 0.01341. Clean and noisy gains were positive on all four development
matches. The composition was declared after contact's external screen, before inspecting spatial
gap results or any target check; it is an additional development attempt, not independent evidence.
No blend coefficient was searched. The externally best passing recipe was selected once.

Spatial training uses only external alive frames with missing ball, fixed noise regimes,
four-second label coarsening, team-swap augmentation, equal provider/match weights and a small
HGB classifier. The three old SkillCorner matches are training/development, never confirmation.
Linear anchors are uncertain geometric features, not claimed recovered ball positions. Every
recipe preserves rule dead **evidence**; decoded D predictions can nevertheless change through
joint temporal decoding.

## Frozen finalist: reserved IDSSE

| Metric | Refitted baseline | contact_spatial | Change |
| --- | ---: | ---: | ---: |
| Clean spellF1 | 0.44747 | 0.48363 | +0.03616 |
| Noisy spellF1 | 0.38329 | 0.42644 | +0.04314 |
| Clean macroF1 | 0.82195 | 0.82753 | +0.00558 |
| Noisy macroF1 | 0.79228 | 0.80142 | +0.00914 |

Per-match noisy gains: J03WPY +0.02403, J03WQQ +0.05608, J03WR9 +0.04933. The descriptive
match-bootstrap 95% interval is [0.02403, 0.05608], but three reused matches do not establish a
new-match population claim; exact two-sided sign-flip p=0.25. More noise seeds do not fix that.

## New-provider corroboration: weak, mixed

Six next-sorted SkillCorner bundles were reserved before V3 scores and acquired only for the
frozen finalist, at commit `4340d274572876239c154c90bc507a9b3250a656`. All files were hash-verified.
Their provider-derived phase labels are **weak algorithmic proxies, not corrected possession gold**.
They were never fitted or used to choose a different recipe.

| Match | Weak-label spellF1 change |
| --- | ---: |
| 1925299 | +0.02887 |
| 1927964 | -0.01962 |
| 1953632 | +0.03119 |
| 1959846 | -0.00587 |
| 1986691 | +0.02932 |
| 1996435 | -0.01881 |
| Equal-match mean | +0.00751 |

Mean macroF1 gain +0.00137; match-bootstrap interval [-0.01075, 0.02400]. This does not establish
strong cross-provider transfer. These six matches are now consumed, not an untouched future holdout.

## Target check: unambiguous rejection under the declared gate

| Diagnostic | Frozen v3c | Frozen finalist |
| --- | ---: | ---: |
| Public spellF1 | **0.60000** | **0.46061** |
| Public macroF1 | 0.82715 | 0.78172 |
| Ten-seed transplant mean | 0.55919 | 0.45403 |
| Transition-centred 5 s | 0.40000 | 0.29677 |
| Transition-centred 10 s | 0.41538 | 0.35862 |
| Transition-centred 20 s | 0.47692 | 0.34286 |

Public predicted possessions increased 11→16 (17 true), but matching the count is not matching
the intervals. Full-match predicted possessions increased 264→329; their accuracy is unknown.
Candidate public 90% interval [0.31515, 0.61818]. The cause of the domain failure is not established
by these aggregate results. No parameters were revised from them.

The official validator exited 0; a separate reload/features/inference/write pass reproduced the
candidate byte-for-byte (SHA-256 `4aa15b6a7104daa8f526d642327ddcd6c973156a73c2563736ee28f896b58f0e`).
Those checks establish format/reproducibility, not acceptance. The original production digest
remains `374f198f07ccecbc5f7b6231235c40eb6bbda3c66c391687bff6de5dc22d7758`.

## Verification and limitations

- Source/mock audit initially exposed incomplete experimental input-mutation checks. Coordinator
  extended them to every Match field, feature keys/content and rebinding, and enforces unchanged
  D evidence. All 36 hardened audit tests pass; no expected failures remain.
- Contact's development scores reproduced exactly under the hardened evaluator. The composition's
  first evaluation used it. Per-candidate synthetic tests verify symmetry; scores do not prove it.
- Earlier contact memos overstated the baseline's minimum reaction time: 12 exclusive frames can
  exceed 0.3 in a 37-frame average. That factual correction changed no recipe or score.
- Target stress evaluates first-half candidate inference after full-match feature normalization;
  synthetic predictions are bit-identical to full inference, and all baseline stress scores match
  the earlier full-pipeline results. This avoids repeated unnecessary local full-match inference.
- Model artifact: 257,589 bytes, trained externally, hash-verified on transfer, kept outside Git.
  All frame-level outputs stay outside the repository. No rejected model replaces v3c.

## Commands and retained research artifacts

Source, declarations and aggregate logs remain in ignored `.orchestration/v3/`; the following
commands describe this checkout, not a promise that scratch code ships in a fresh clone:

```sh
# RunPod only; external data, fixed development/confirmation split:
scripts/py .orchestration/v3/evaluate.py --module .orchestration/v3/composed/candidate.py --out .orchestration/v3/composition-development.json
scripts/py .orchestration/v3/evaluate.py --module .orchestration/v3/composed/candidate.py --recipe contact_spatial --mode confirmation --out .orchestration/v3/composition-confirmation.json
scripts/py .orchestration/v3/confirmation_followup.py
# Local lightweight inference of the transferred, frozen artifact:
scripts/py .orchestration/v3/artifacts.py pull
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 scripts/py .orchestration/v3/target.py native
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 scripts/py .orchestration/v3/target.py stress
```

All substantial jobs ran through the visible managed RunPod lease, with aggregate-only logs.
The lease also supported the separately declared V4 observation-source investigation, then was
deleted (exit 0; independent owned-pod inventory empty). Successful pod runtime was 2.56 h at
$0.24/h, about $0.61 compute, not a reconciled invoice. PLAN records cleanup and limitations.
No commit or push was performed.
