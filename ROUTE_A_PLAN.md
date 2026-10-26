# Route A — possession via geometric features + structured temporal decoding

Status: **proposed** (pending dependency decision, see *Dependencies*). Date: 2026-10-26.
Deadline: **2026-11-17** (submission freeze: **2026-11-14**).
This is the implementation plan for the score-first route chosen over a language-model route.
On approval, tasks merge into `docs/PLAN.md` and acceptance criteria into `docs/SPEC.md`
(spec-driven loop); this document stays as the reference with the full rationale.

## Quick path

1. **Phase 0 starts today** — it needs **no match data**: public-GT forensics (B1), DFL ingestion, feature module, leave-one-match-out training.
2. Phases 1–2 run as soon as the match data lands (H1): ATD audit, deterministic baseline submission, full model.
3. Freeze on 2026-11-14: one command regenerates the submission byte-for-byte, `validate_submission.py` passes, technical document written.

## Objective and guardrails

**Goal**: beat the full-match baseline floor, smooth50 spellF1 **0.2433** [EVALUATION.md:105-116], with a
deterministic pipeline: identity-free geometric features -> calibrated per-frame posteriors -> Viterbi decoding,
pretrained on external per-frame possession data, calibrated on the public 5-minute ground truth.

**Honest measurement rule**: we cannot measure full-match spellF1 locally (no full GT). Therefore:
model selection happens **only** on external validation (leave-one-match-out on DFL); the public 5-min GT is used
**only** for calibration with `--bootstrap` intervals; full-match sanity checks are aggregate-only
(possession counts, mean duration). The public window demonstrably misranks baselines — smooth125 wins public
(0.4242) and loses the full match [EVALUATION.md:130-137] — so it must never drive model selection.
Noise floors: <0.01 macroF1, <0.04 spellF1 [EVALUATION.md:43-44].

**Guardrails (from the rules, verified)**:

- Labels in the submission must come from an algorithm [RULES.md:39]; small hand-labeled *training/validation*
  data is allowed [RULES.md:47-48]; any corrected possession data for this fixture is banned [RULES.md:52-54].
- External data, pretrained models and LLMs are explicitly allowed [RULES.md:12-14]; organisers must be able to
  access and inspect everything; the technical document must explain the pipeline [RULES.md:9-10].
- The repository must regenerate the submission **byte-for-byte** [RULES.md:56-59].
- Match data never reaches a cloud LLM; scripts print aggregates only [docs/DATA_POLICY.md; OVERVIEW.md:150-155].
- No data, weights, or frame-level outputs in Git; new dependencies are a proposal first [AGENTS.md].

## Why this route (evidence)

- Nothing in the official docs requires an LLM; "the use of LLMs is not itself an evaluation criterion this year"
  [OVERVIEW.md:157-158]. The task is framed as a tracking/geometry problem [CHALLENGE.md:107-121].
- Closest public precedent: **PathCRF** (KDD 2026, MPL-2.0, checkpoints public), trained on the same DFL data —
  masked CRF + Viterbi gave ~+10pp event precision over per-frame argmax with 0% illegal transitions.
  Temporal structure matters far more than model scale.
- Small-data regime: 7 pro matches of external labeled tracking + 5 minutes of public GT favor classical
  structured models over deep nets.
- "Positions over pixels": the broadcast video is not synchronised; in SoccerNet-GAR a 180K-parameter
  Tracking-GNN beat video baselines. Tracking features are the primary signal; video is out of scope for Route A.

## Design decisions

| # | Decision | Why |
|---|----------|-----|
| 1 | Identity-free features (positions + team only) | The ATD has no player identities, only team membership; DFL/SkillCorner have identities. Features computable from team-labeled positions run unchanged on all three sources — this is the transfer guarantee. |
| 2 | Temporal windows defined in seconds, not frames | DFL and MTAH are 25 fps, SkillCorner 10 fps; second-based windows make features frame-rate invariant. |
| 3 | Ball-dropout augmentation (~40%, seeded) on DFL | The ATD ball is missing 43% of frames [DATA.md:107-115]; DFL rarely misses. Synthetic dropout closes the domain gap deterministically. |
| 4 | Calibrated softmax posteriors + Viterbi (HMM-style) with transitions learned from DFL H/A/D sequences | The PathCRF lesson: structure >> scale. Numpy-only core: deterministic byte-for-byte, no GPU nondeterminism, tiny dependency surface. |
| 5 | Model selection only on leave-one-match-out (LOMO) over DFL, scored with the repo's own spellF1 | Same metric as the competition; avoids the documented public-window overfit trap. |
| 6 | D detection trained from DFL `BallStatus` (0=dead), anchored by Smart Tagging restarts; hard rule for the pre-kickoff block | macroF1 (tie-break) averages H/A/D, so D quality counts. 2,934 pre-kickoff frames of the second half are all D [FAQ.md:33-35]; every set piece is a restart ending a dead-ball stretch [DATA.md:169-171]. |
| 7 | `X` only inside stable spells | `X` is forward-filled ("repeat my previous answer") [possession_metrics.py:274-293]: abstaining is safe mid-spell and costly at boundaries, where it forfeits the new spell. |

## Architecture (per docs/CODING.md layout)

| Module | Role |
|--------|------|
| `possession/features.py` | Pure functions: common frame table -> feature matrix. Synthetic tests beside it. |
| `possession/io_dfl.py` | DFL loader -> common frame table (uses floodlight, or fallback parser). |
| `possession/io_skillcorner.py` | SkillCorner loader -> same frame table. |
| `possession/atd.py` | Thin adapter over the upstream/kloppy ATD loaders -> the same frame table. |
| `possession/model.py` | Softmax regression (numpy or sklearn) + Viterbi decoding with learned transitions. |
| `possession/postprocess.py` | X policy, dead-ball stretches, pre-kickoff block, hysteresis. |
| `possession/train.py` | LOMO training entry; prints aggregates only. |
| `scripts/fetch_external.py` | Pinned downloads (DOI + SHA256) into `~/.mtah2027/external/` (outside Git). |
| `run.py` | Single command regenerating the submission: `scripts/py run.py --out evaluation/results/<name>.csv`. |

Common frame table carries units, team side and frame base per CODING.md ("carry meaning in the data").
Every phase ends with a valid submission — the pipeline is never left in a broken state.

## Phases and tasks

### Phase 0 — no match data needed (starts immediately)

| ID | Task | Acceptance criteria |
|----|------|---------------------|
| T1 (=B1) | Public-GT forensics | `docs/research/` notes: the 17 public possessions (durations, D stretches, hand-overs) and how spellF1 reacts to fragmentation, smoothing and X on synthetic series through the official scorer. |
| T2 | DFL ingestion | `scripts/fetch_external.py` downloads the pinned figshare record with SHA256 check; loader produces the frame table; a script prints aggregates only (frames, H/A/D balance per match); minimal synthetic XML fixture test. |
| T3 | Feature module | `possession/features.py` passes synthetic tests for the full CODING.md fixture list: missing ball, track gaps, identity switches, dead-ball stretches, half boundaries, single-frame flicker. |
| T4 | LOMO training | Softmax + Viterbi trained on 7 DFL matches with ball-dropout augmentation; `docs/research/` table with spellF1/macroF1 per held-out match, computed with the repo scorer. |

### Phase 1 — needs the match data (H1)

| ID | Task | Acceptance criteria |
|----|------|---------------------|
| T5 (=B2) | ATD audit (aggregates) | Answers the open question (does the ATD carry alive/ownership fields?); prints ball visibility per half, track-slot counts, Smart Tagging class counts, homography validity. No rows. |
| T6 (=B0) | Reproduce baselines | `scripts/py starter/baseline.py` regenerates the four baselines; public scoreboard reproduced. |
| T7 | ATD adapter | Full-match frame table; aggregate statistics only. |
| T8 (=E1a) | Deterministic baseline (ours) | Nearest-player + hysteresis + smoothing -> first own submission: passes `validate_submission.py`; public spellF1/macroF1 with bootstrap. This is the safety-net submission. |

### Phase 2 — the model

| ID | Task | Acceptance criteria |
|----|------|---------------------|
| T9 (=E1b) | Full model on the match | Softmax + Viterbi over the full match, thresholds calibrated on the public GT; candidate submission + aggregate sanity report (possession count, mean duration — CODING.md reporting format). |
| T10 (=E2) | X + D + pre-kickoff policy | Ablation on the public GT with bootstrap intervals for X policy, dead-ball stretches, pre-kickoff rule. |
| T11 (=E3) | Smart Tagging priors | Set pieces -> dead-ball stretch seeding; team tags -> prior feature; ablation. |

### Phase 3 — robustness and freeze

| ID | Task | Acceptance criteria |
|----|------|---------------------|
| T12 (=E4) | SkillCorner mix | Frame-rate-invariant features; ablation on external LOMO validation. |
| T13 (=E5) | Freeze | One command reproduces the submission byte-for-byte (seeds, pinned deps, checksummed downloads); `validate_submission.py` passes; technical document written [RULES.md:9-10]. |
| T14 (stretch) | Small TCN | Only if external validation shows a persistent gap and time allows; determinism still enforced. |

Every task closes per the CODING.md done checklist (pure logic tested, tests pass, no data printed,
PLAN.md records the evidence) and lands as a small purposeful commit with its tests and doc updates.

## Timeline (deadline 2026-11-17)

| Week | Window | Scope |
|------|--------|-------|
| 1 | 2026-10-26 -> 11-02 | Phase 0 complete; H1 on the team side; if data arrives, T5–T8 including the safety-net submission. |
| 2 | 11-02 -> 11-09 | Phase 2 complete: model submission, X/D policy, Smart Tagging priors. |
| 3 | 11-09 -> 11-14 | Phase 3 + freeze. 11-14 -> 11-17: buffer and submission only. |

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Domain shift DFL -> ATD (no identities, 43% missing ball, league style) | Identity-free features, ball-dropout augmentation, calibration on public GT; fallback is T8 (same family as the 0.2433 floor). |
| ATD carries no alive/dead field | D from the model + Smart Tagging restarts + ball-absence kinematics + pre-kickoff rule. |
| Overfitting the 5-minute public GT | Selection only on external LOMO; public GT for calibration only; bootstrap intervals; noise floors. |
| Reproducibility | Numpy-only core (no torch/GPU); fixed seeds and stable sorts; checksummed pinned downloads. |
| Time | Incremental valid submissions each phase; T14 only as stretch. |

## Dependencies (decision pending — AGENTS.md requires approval)

| Package | Purpose | Alternative |
|---------|---------|-------------|
| `floodlight` (pip) | DFL XML loader with per-frame `BallPossession`/`BallStatus` (the loader cited by the dataset paper; depends on numpy/pandas already present) | Hand-rolled DFL XML parser — more code, format risk. |
| `scikit-learn` (pip) | Softmax regression + calibration | ~100 lines of numpy; zero new dependencies. |
| — | Check first whether `kloppy` (already installed) already parses DFL | If yes, drop `floodlight`. |

External data lives in `~/.mtah2027/external/` (outside Git), fetched by a pinned script.

## Evidence and sources

Repo (public): `CHALLENGE.md` (task, labels 22-25), `EVALUATION.md` (floor 105-116, misranking trap 130-137,
noise floors 43-44), `RULES.md` (method freedom 12-14, algorithm labels 39, hand-label training 47-48,
private GT ban 52-54, reproducibility 56-59, technical document 9-10), `OVERVIEW.md` (LLM terms 150-155,
LLM not scored 157-158), `DATA.md` (ATD 68-115, Smart Tagging 122-179, weak-prior note 169-171),
`FAQ.md` (pre-kickoff D block 33-35), `docs/PLAN.md` (task states), `docs/CODING.md` (layout, tests, fixtures).

External (verified 2026-10-26):

- **DFL/Sportec — Bassek et al. 2025**, "An integrated dataset of spatiotemporal and event data in elite soccer",
  Sci Data 12:195, CC BY 4.0, figshare DOI 10.6084/m9.figshare.28196177: 7 Bundesliga 1/2 matches (2022/23),
  TRACAB 25 Hz, 1,002,644 frames; per frame on the ball stream `BallPossession` (1=home, 2=away) and
  `BallStatus` (0=inactive, 1=active); loadable via `floodlight.io.dfl`; no video.
- **SkillCorner opendata**, https://github.com/SkillCorner/opendata, MIT: ~20 A-League 2024/25 matches
  (README says 10; the repo holds 20 match folders), 10 fps broadcast tracking with a per-frame `possession`
  field, dynamic events, phases-of-play.
- **Metrica sample-data**, https://github.com/metrica-sports/sample-data: 3 matches, 25 fps, events
  synchronised to frames (Start/End Frame columns); informal license (acknowledge source). Same provider
  family as this hackathon. No possession labels.
- **StatsBomb/Hudl open data**, https://github.com/hudl/open-data: 4,235 event matches with
  `possession`/`possession_team` fields (event-level priors only; LICENSE.pdf to verify before use).
- **PathCRF**, https://github.com/hyunsungkim-ds/pathcrf, MPL-2.0 with checkpoints (KDD 2026,
  arXiv 2602.12080) — closest public precedent; trained on the same DFL dataset.
- **LaurieOnTracking** (pitch control / EPV feature references), MIT;
  **Ball Radar** (rule-based baseline); **dude.k** / **T-DEED** (winning spotting patterns: weak-label
  pretraining -> dense fine-tune, joint team-action heads, horizontal flip with team swap, Soft-NMS).
- Refuted: no SoccerNet "Ball Possession" challenge exists in any edition through 2027.

## Next step

Approve the dependency decision (floodlight + scikit-learn, or numpy-only variant). Then merge T1-T14 into
`docs/PLAN.md`, update `docs/SPEC.md` acceptance criteria, and start Phase 0 — it requires no match data.
