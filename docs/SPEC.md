# MTAH 2027 — spec

## Goal

Produce `evaluation/results/<name>.csv`: `frame,label`, 199,319 rows (one per tactical-video frame),
labels `H` (Barcelona in possession, ball alive), `A` (Athletic), `D` (dead ball), `X` (abstain).
Ranked on **spellF1** over the private full-match ground truth, tie-break **macroF1**.
Floor to beat: `baseline_smooth50`, spellF1 **0.2433** on the full match.

## Deliverables (SUBMISSION.md)

1. The possession file, validated with `evaluation/validate_submission.py`.
2. A **public** repository (this one, pushed to our own remote) that regenerates the file
   byte-for-byte from the provided data with one documented command, with pinned dependencies.
3. A short technical document: approach and why, assets used and dropped, handling of missing and
   noisy tracking, possession rules, limitations, how to reproduce.

## What we know about the ground truth (CHALLENGE.md)

- Derived mechanically from **corrected** tracking: `D` where it marks the ball dead, otherwise the
  team it names as owning the ball. No smoothing. The public 5 minutes (17 possessions, 7,500 graded
  frames) show what "dead" and "owning" mean in practice.
- A possession is a maximal run of one team with the ball alive. Contested or loose balls stay with
  the last team in possession. A touch, a deflection or an aerial duel without control is not a change.
- `X` on graded frames is a miss for macroF1 and "repeat your previous label" for spellF1.

## Inputs (DATA.md; gated, see DATA_POLICY.md)

Tactical video (reference timeline) and broadcast video (not synchronised), automatic tracking with
the ball for both (team-level, 46 slots, no player ids, ball missing on 41% of graded frames),
Smart Tagging (638 automatic instances), homography per frame.

## Method constraints

- Labels are produced by an algorithm only (RULES.md §1).
- No ground truth beyond the public 5 minutes.
- Any model or service in the pipeline must be accessible to the organisers. **Hosted LLMs never see
  the data** (DATA_POLICY.md), so any LLM inside the pipeline is local.
- Deterministic: same data in, same file out.
- Data-safety rules are shared; active agent configuration (`.pi/`, `.claude/`) stays local and
  ignored. Reusable guard code and tests remain optional tooling in the repo.

## Open questions

- Team members (3–5 needed for the Innovation Challenge and Final Pitch eligibility).
- Name and owner of the public submission repo.
- Compute budget: everything local on the M3 Pro (18 GB) unless we decide otherwise.
