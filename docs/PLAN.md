# MTAH 2027 — current execution plan

Updated: 2026-10-26

Status vocabulary: `ready`, `in progress`, `blocked`, `done`.
Only mark done when acceptance evidence is recorded. Evidence is aggregate only (DATA_POLICY.md).

| ID | Status | Task | Acceptance criteria |
| --- | --- | --- | --- |
| H0 | done | Working environment and agent harness | Official repo cloned (`upstream`), data root `~/.mtah2027` outside the repo, data guard for pi and Claude Code with passing tests, project Conda env, AGENTS/CLAUDE/docs in place. Evidence: guard tests 12/12 OK; env import check (see H0 notes). |
| H1 | blocked on user | Get the match data | User enrols, accepts terms, downloads, and unpacks into `~/.mtah2027/data` keeping the names in `data/README.md`. `scripts/data-status.sh` shows all tactical files `ok`. |
| B0 | blocked on H1 | Reproduce the baselines | `scripts/py starter/baseline.py` regenerates the four baselines; `scripts/py evaluation/run_evaluation.py` reproduces the README scoreboard on the public 5 minutes. |
| B1 | ready | Understand the scorer and the public ground truth | From public files only: summarise the 17 public possessions (durations, `D` stretches, hand-overs) and how spellF1 reacts to fragmentation, smoothing and `X`. Notes in `docs/research/`. |
| B2 | blocked on H1 | Data audit (aggregate only) | A script prints coverage statistics: ball visibility per half, track slot counts, any ownership/alive fields in the ATD, Smart Tagging class counts, homography validity. No rows printed. |
| E1 | done (external) | First method beyond nearest-player smoothing: v3 (`possession/`, `run.py`) | Hypothesis: two-stage Viterbi with learned dead-ball and control/shape evidence beats window voting. Evidence: leave-one-game-out on Metrica's public sample games, spellF1 0.544 vs 0.247 for smooth50, paired Δ +0.298 [+0.286, +0.312] (`docs/research/v3-external-validation.md`). Tests: `python -m unittest discover -s possession -p 'test_*.py'` → 5 OK (8 OK with v4). |
| E2 | done (local, 2026-10-25) | Run v3 on the match | Evidence (aggregates, Ángel's machine): reader matches DATA.md (199,319 frames, 46 slots, ball seen on 59.3 % of graded frames); `--baseline 50` identical to `baseline_smooth50.csv`; v3 output valid, same SHA-256 on two runs; public 5 min (sanity only) spellF1 0.3412 / macroF1 0.7599 / 17 possessions; full match 220 possessions, D 61.2 % of graded frames, i.e. stoppages over-called (others publish 35–47 %). Led to E3. |
| E3 | done (external + local 2026-10-26) | v4: dead ball robust to the tracker (`config/possession_v4.json`, default in `run.py`) | Evidence: Metrica leave-one-game-out mean over 7 test-noise scenarios 0.536 vs 0.499 (v3); SkillCorner paired v4 − v3 +0.036 [+0.030, +0.042]; rehearsal 0.581 vs 0.546. Local (aggregates): valid, same SHA-256 twice, v3 unchanged; 223 possessions (v3 220), mean 14.1 s, D 50.4 % (v3 61.2 %); gap report recorded; public 5 min sanity 0.5733 / 0.8669 (`docs/research/v4-robust-dead-ball.md`). |
| X2 | parked | Smart Tagging as evidence beyond the kick-offs | `tag_report.py` on the match (aggregates): the defending codes mirror the attacking ones; build up / progression agree with confident v4 99.7 % / 94 % but cover only 1–4 % of ball-gap frames; the codes that cover gaps agree 64–78 %; set-piece tags start while the ball is already dead and do not mark the restart. Low expected gain. Possible small test: change-point prior at attacking-transition starts (v4 hands the ball to the tagged side within 3 s in 38 % of 21 cases). |
| E5 | done, not kept | Turnover-aware switching cost; benchmark labels closer to "controlled possession" | Per-frame team switching cost: Metrica −0.001..+0.001, SkillCorner +0.0006 [−0.0013, +0.0022]. Controlled-possession labels: v4 still beats v3 (+0.010 [+0.003, +0.017]); retraining on them −0.004. Details in the v4 research note. |
| E6 | done (2026-10-26) | v5: v4 trained on Metrica + Sportec DFL (`config/possession_v5.json`, default of `run.py`) | DFL matches held out +0.009 [+0.005, +0.013] over v4; SkillCorner +0.011 [+0.008, +0.015]; Metrica held out −0.012 [−0.019, −0.005] (fragmented event labels). Local check passed: valid, same SHA-256 twice, 221 possessions (v4 223), D 51.5 % (v4 50.4 %); public 5 min sanity 0.5733 / 0.8492 (`docs/research/v5-dfl.md`). |
| U1 | ready | Sync upstream (23 Oct rule clarifications) | `git fetch upstream && git merge upstream/main`. Upstream now says the public 5 minutes are not for training or tuning and that external data must be about other games. AGENTS.md "Tune against them" must be aligned with that. |
| X1 | done (2026-10-26) | Third external set and training data: Sportec Open DFL (7 matches, per-frame BallStatus / BallPossession from corrected tracking, CC-BY 4.0, data © DFL) | Downloaded and converted locally with `scripts/fetch_dfl.py` (kloppy 3.19.1, Hugging Face mirror). Labels smoothed to "controlled possession" (`dfl.controlled`, ≥ 2 s). Benchmark: v4 − v3 +0.008 [+0.003, +0.013]; training with it: E6. `docs/research/v5-dfl.md`. |
| E7 | done (2026-10-27) | Blind head-to-head of the team pipelines on 20 unseen SkillCorner matches laid out as challenge bundles (each pipeline's own `run.py`, official scorer; `scripts/rehearsal.py`) | spellF1: v5 0.351, main (Fran, 60d0bfc) 0.331, Route A (Javi, baf223b, tags off) 0.315, b50 0.233. v5 − main +0.021 [+0.011, +0.031], ahead on 18/20; v5 − Route A +0.036 [+0.027, +0.046], 20/20. The gap is mostly dead ball: D share 43 % / 22 % / 20 % against 51 % in the truth; Route A also fragments (416 possessions for 293). `docs/research/head-to-head-skillcorner.md`. |
| B3 | done, no-go as is; parked (2026-10-27) | Ball-detector pilot on the tactical video: does a COCO-pretrained detector find the ball in the ATD's gaps? (`scripts/ball_detector_pilot.py`, aggregates only) | Hypothesis: the ball is visible on part of the ATD's in-play gaps; simulated value +0.016 spellF1 if in-play gaps halve, +0.035 at 5 % missing. Go criteria, set beforehand: where both see the ball ≥ 80 % within 2 m; fills ≥ 25 % of 1–10 s gap frames with consistent detections; best video/ATD offset 0; whole match ≤ 12 h on the local machine. If go: extra ball channel (ATD ∪ consistent detections), determinism check, value re-simulated with the measured fill and precision. Rules: own detector allowed; no hand judging of frames. **Result** (local, RTX 5060, yolo26x, 1280 tiles, 3,000 frames from each kick-off area: 11185 and 130000): video/ATD offset 0 on both; 5.4 frames/s (whole match ~8.2 h); where both see the ball the top detection is within 2 m on 59 % / 53 % of frames and beyond 10 m on 23 % / 28 % (confidence ≥ 0.5: ~70 % within 2 m); ~2 detections per frame, so the gap "fills" (96 % of 1–10 s gap frames, 90 % of a ≥ 10 s gap, only 43 % near a player) cannot be trusted. Fails the agreement criterion. Parked: a tracker choosing among the candidates could be tested by hiding the ATD ball on seen stretches, but the upside is ≤ +0.016, real gaps are harder than seen frames, and GPU output may not regenerate byte for byte (RULES rule 3). |

## H0a — local agent configuration

Status: done. User-requested cleanup of tracked developer-tool settings.
Acceptance: `.pi/` and `.claude/` are ignored and untracked; existing local configurations remain
unchanged; shared rules and reusable guard code/tests remain tracked; docs do not promise guards
on fresh clones. Preserve the existing `.gitignore` edit. Verify Git state and guard tests.
Evidence: `git ls-files .pi .claude` empty; `git check-ignore` confirms both configurations ignored;
`cmp` against HEAD confirms both local files unchanged. `git diff --check` passed;
`python3 -m unittest discover -s scripts -p 'test_*.py'` → 12 tests OK.
No hook code changed; runtime hook behavior was not re-tested. No commit or push performed.

## H0 notes

- Guard tests: `python3 -m unittest discover -s scripts -p 'test_*.py'` → 12 tests OK.
- pi loads `.pi/extensions/data-guard.ts` only after the project is trusted. End-to-end: a pi run with the extension blocked `read` of a dummy `.mp4` with the DATA GUARD message.
- Env: `.conda-env` Python 3.11.17, numpy 2.4.6, pandas 3.0.6, kloppy 3.18.0.
- `scripts/py evaluation/run_evaluation.py` reproduces the README public-5-min scoreboard exactly (smooth125 0.4242 / smooth50 0.3256 / smooth25 0.2868 / raw 0.0378 spellF1).
