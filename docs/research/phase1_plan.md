# Phase 1 plan, explained from zero

Status: **draft for team review** (updated 2026-10-22, reviewed against the scorer and the upstream
docs). The match data is downloaded (all nine assets present, `scripts/data-status.sh`), but no
script has read it yet, so every score in this document is still an **estimate**, not a
measurement. Numbers marked *(measured)* were computed on the public 5 minutes with the official
scorer; numbers marked *(other team)* are second-hand and get replaced by our own measurement in W0.

This document is written for teammates with little or no football or machine-learning (ML)
background. Read it in order: each part builds on the previous one.

1. Football basics needed for this task
2. The task: what we must produce
3. How our answer is scored
4. The data we receive
5. The biggest problems
6. Training, validation and test: what each one is for us
7. Our approach, step by step
8. How we will know we are improving, and what we cannot know
9. What we hand in, and the rules about it
10. Plan, team split, budget
11. Chances, risks, open questions
12. Glossary

---

## 1. Football basics needed for this task

You only need five ideas.

- **Two teams of 11 players** play with one ball. Here: **FC Barcelona** (the *home* team, letter `H`)
  against **Athletic Club** (the *away* team, letter `A`). A match is two halves of about 45 minutes,
  with a break (*halftime*) in between.
- **Ball in play ("alive") vs out of play ("dead").** The ball is *alive* while the game is running.
  It is *dead* when the referee has stopped play: the ball went over a line, there was a foul, a
  goal, a substitution, an injury. While it is dead, players wait or walk; then play *restarts*
  (throw-in, corner kick, free kick, goal kick, kick-off). In the 5 minutes of answers we have, the
  ball is dead about 20% of the time.
- **Possession** = which team has the ball while it is alive. The rules of this challenge say:
  while the ball is alive, **one team always has possession**. There is no "nobody" state.
- **Possession changes only on control.** If an opponent just touches the ball, deflects a pass or
  wins a header but his team does not keep the ball, possession does **not** change. If the ball is
  loose or in the air, possession stays with **the last team that had it**. It changes when a player
  of the other team actually gets the ball under control.
- **A possession** is one uninterrupted period of one team having the ball. Example, one letter per
  moment: `H H H A A D D H H` = three possessions (Barça, then Athletic, then after the stoppage
  Barça again). The dead-ball part `D` is a pause, not a possession.

Typical rhythm, from the 5 minutes of answers we have: 17 possessions in 5 minutes, average 14
seconds, but some are very short (one lasts 0.8 seconds, six last under 4 seconds).

## 2. The task: what we must produce

The match was filmed by a fixed wide camera (the *tactical* video). A video is a sequence of still
images called **frames**: 25 per second. The full recording has **199,319 frames** (about 2 h 13 min,
including before the match, halftime and after).

For **every frame** we must write one letter:

| letter | meaning |
|---|---|
| `H` | ball alive, Barcelona has it |
| `A` | ball alive, Athletic has it |
| `D` | ball dead (game stopped) |
| `X` | "I don't know" (allowed; see section 3) |

The output is a simple table (a CSV file) with 199,319 rows: `frame number, letter`.

**Important rule:** the letters must be produced by a program. Nobody may watch the video and type
the answers. Section 9 explains everything we hand in.

## 3. How our answer is scored

The organisers have the correct letter for every frame (the **answer key**, which they call *ground
truth*). They keep it secret and compare our file against it after the deadline. There is no
leaderboard during the competition.

Only the frames of the two halves are graded: 158,857 frames (about 106 minutes). Before kick-off,
halftime and after the final whistle do not count.

### The ranking score: spellF1 ("did you find the real possessions?")

The scorer does not mainly count correct frames. It cuts both files into possessions and asks, for
each real possession, *did you predict one that overlaps it well, with the same team?* "Well" means
the overlap is a large share of the two together. Roughly: found real possessions and invented fake
ones both count. 1.0 is perfect; 0 means nothing found.

Why this matters, with numbers we measured on the 5 public minutes (taking the correct answers and
damaging them on purpose):

| What we damage in a perfect answer | Score |
|---|---|
| nothing | 1.00 |
| every possession change predicted 3 frames (0.12 s) late | 1.00 |
| every change 8 frames (0.3 s) late | 0.95 |
| every possession change predicted 0.5 s late | 0.91 |
| every change 1 s late | 0.78 |
| the two possessions shorter than 2 s missed | 0.83 |
| the six short possessions (< 4 s) missed | 0.55 |
| **every dead-ball stretch missed** (labelled as the previous team) | **0.81** (but macroF1 0.59) |
| one single wrong *dead-ball* frame in the middle of every possession | 0.31 |
| **one single wrong frame in the middle of every possession** | **0.24** |

All *(measured)* with the official scorer on the public window; the script is a throwaway, the
numbers are reproducible from `evaluation/possession_metrics.py` and the public ground truth.

The last line is the key lesson: **a single "flicker"** (one frame giving the ball to the wrong team)
cuts one real possession into three pieces, and the scorer counts that as a miss plus two invented
possessions. A program that is right on 99.9% of frames can still score terribly. So our output must
be **calm** (no flickering) but still **quick** enough to catch short real possessions. That
balance is the whole game.

Two more lessons from the table that set priorities:

- **Hand-overs between teams are what the ranking score pays for; dead balls are second.** Missing
  *every* dead-ball stretch costs 0.19 on the ranking score (it only hurts where the same team has
  the ball before and after the stoppage) but 0.41 on the tie-break score. So the order of work is:
  (1) place the H↔A changes precisely, (2) never flicker, (3) then dead balls, which also buy
  macroF1. Do not spend week 1 on dead-ball detection.
- **Boundaries must be precise to a handful of frames.** A 3-frame error is free, 8 frames already
  cost 0.05, because the scorer matches by overlap and a 1-second possession shifted by 0.3 s no
  longer overlaps enough. We therefore decode at full 25 fps and never smooth with a window that
  delays boundaries; the storyteller (section 7, step 3) looks ahead, so it has no built-in lag.

The second score, **macroF1**, counts correct frames per letter. It is only used to break ties.

**About `X` ("I don't know"):** for the ranking score, an `X` is replaced by our previous letter.
So `X` in the middle of a possession is harmless; `X` across a real change of possession is as bad
as being wrong. In practice we will rarely use it.

**The bar to beat:** the organisers' simple example program (nearest player to the ball, then
smoothing) scores **0.2433**.

## 4. The data we receive

We do not have to analyse raw video to get positions; the organisers give us processed data.

| Data | What it is, in plain words | How reliable |
|---|---|---|
| **Tracking data** | For every frame: the position (x, y on the pitch) and speed of each player that a computer saw, and of the ball. Each player is marked as Barça or Athletic. | Automatic, **not corrected**. Has holes and errors. |
| **Smart Tagging** | 638 automatic notes like "Barcelona build-up from minute 3:10 to 3:25", "corner kick", "throw-in". | Automatic guesses; sometimes wrong; phases are not possessions. |
| **Homography** | A formula per frame to convert pitch positions to pixels in the video and back. | Good. Only needed if we use the video. |
| **Videos** | The tactical video (our reference) and the TV broadcast (not aligned in time). | Raw images. Heavy to process. |
| **5 minutes of answers** | The correct letters for the first 5 minutes of the match (frames 11,185 to 18,684): 17 possessions. | Correct. Public, in our repo. |

Facts about the tracking that shape everything:

- **No player identity.** We know "a Barça player is here", not "this is Pedri". A player's track
  can break and continue under another number.
- **The ball is missing in 41% of the graded frames.** The computer simply did not see it (hidden
  behind players, in the air, in a crowd in front of goal).
- **The tracking was computed by a computer from the same tactical video.** Where it lost the
  ball, the ball is usually really hard to see in the video too.
- The answer key was made from a **corrected** (cleaned-up) version of this tracking. We get only the
  uncorrected version. So the answer key "knows" things our data does not show.

## 5. The biggest problems

**Problem 1: the ball is often invisible.** On 41 of every 100 graded frames there is no ball
position (organisers' figure). The gaps are long (typical gap about 11 seconds, longest 3.7 minutes
*(other team)*; W0 re-measures this). Think of listening to a match on a radio that cuts out for 11
seconds at a time.

One thing the organisers gave us that the plan did not use so far: the **broadcast camera has its
own tracking with its own ball**, visible on 50.5% of *its* frames (OVERVIEW.md), from a different
viewpoint. It is on a different clock, with cuts and replays, and no mapping is provided (FAQ). But
both trackings give player positions on the same pitch coordinates, so the two clocks can be
aligned by matching player formations frame to frame, with no identities needed, and replays
reject themselves because the formations do not match. Whether the broadcast ball fills the
tactical gaps is an empirical question (the two cameras may lose the ball at the same moments),
and it is answered by one aggregate number: *the share of tactical gap frames on which the aligned
broadcast tracking sees the ball*. This costs about two days, needs no model, no weights and no
GPU, and is tried **before** the video detector (section 7, step 4).

**Problem 2: even when we see the ball, "who has it" and "is it dead" are not obvious.** The ball
being visible is not the same as knowing the answer:

- *Who has it:* "nearest player to the ball" names the right team only about 78% of the time when
  the ball is visible (another team's measurement). A pass in flight, a duel, a touch without control
  all fool it.
- *Is it dead:* the tracking has no "game stopped" signal. We must guess stoppages from behaviour
  (players slowing down, ball over a line). The organisers' example program is worst at this.

**Problem 3: we have almost no examples of correct answers.** Only 5 minutes, i.e. 17 possessions.
That is far too few to *learn* from (another team tried; it failed badly) and far too few to *judge*
which version of our program is better: one possession more or less moves the score by about 6
points.

**Problem 4: those 5 minutes are not representative.** In them, the ball is visible almost all the
time (missing only ~12%, versus 41% over the full match). They test the easy situation, almost never
the hard one. The organisers prove the danger: tuned on these 5 minutes, their example program picks
the version that is worse on the full match.

**Problem 5: the scoring punishes flicker hard** (section 3), while the true answers contain some
very short possessions. We must be stable without being slow.

### Is the missing ball *the* problem? Partly.

How much each problem costs, from what we know today:

- **Problems 2 and 5 exist even with a perfect ball.** In the public 5 minutes the ball is visible
  ~88% of the time, almost a "perfect ball world", and the best published approach still scores only
  0.50–0.61 there, not 0.90.
- **The missing ball clearly costs a lot too.** When another team hid the ball on those 5 minutes
  on purpose, their score fell from ~0.61 to 0.35–0.53. That gap is roughly the most a perfect ball
  fix could win back.
- **Most frames inside a gap are easy.** When the ball disappears, simply keeping the last owner is
  right on ~94% of those frames (another team's measurement). The damage is concentrated in the
  possession changes that happen *inside* gaps, which we then miss or catch late.
- **Finding the ball in the video is not guaranteed to work**, because the tracking already came
  from that video (section 4). Where the computer lost the ball, it is often truly hidden.

Conclusion: we attack both fronts. The main line works on problems 2–5 with the data we have; a
parallel line tests early whether the ball can be recovered from the video (section 7, step 4).

## 6. Training, validation and test: what each one is for us

These three words come from ML. Use the analogy of a student preparing an exam.

| Term | Analogy | General meaning |
|---|---|---|
| **Training set** | the exercises with solutions you study from | examples with correct answers that the program *learns* from |
| **Validation set** | practice exams you take to check yourself | examples with answers the program did *not* learn from, used to choose between versions |
| **Test set** | the real exam, marked by the teacher | the data that gives the final grade; you never see the answers |

### Where our "exercises with solutions" come from

There is a free public dataset of **7 full professional German league (Bundesliga) matches**
(*IDSSE / Sportec Open DFL*, Bassek et al. 2025, free licence). It is **not** just a list of answers.
For each match it contains:

| Part | What it contains | Do we have the same for our match? |
|---|---|---|
| **Tracking data**, 25 frames per second | Position and speed of all 22 players and the ball, every frame. Clean: players have identities, the ball is almost never missing. | **Yes**, the same kind of data, but ours is noisy (no identities, broken tracks, ball missing 41%). |
| **Two flags in every frame** | "ball alive or dead" and "which team owns the ball" | **No: this is exactly what we must predict.** These flags are the answers (labels). |
| **Event data** (typed by human operators) | passes, shots, fouls, set pieces, with time and team | Not the same. Our Smart Tagging is a different, automatic product. |
| Video, homography | not included | We have them. |

So every frame of those matches is an exercise of exactly our type: **input = tracking around that
frame; solution = H, A or D.** We verified the meaning of the two flags in the published paper and in
the code of the library we use to read tracking (`kloppy`). Not yet verified: we have not downloaded
the files ourselves (the site needs a browser download). One weakness: **5 of the 7 matches involve the
same team** (Fortuna Düsseldorf), so there is less variety than "7 matches" suggests.

### Our three sets

| Set | What it is concretely | Has answers? | What we use it for |
|---|---|---|---|
| **Test** | The full Barça–Athletic match, graded by the organisers | Yes, but **secret** | Our final score. We see it only after the deadline. |
| **Training** | The 7 Bundesliga matches, **damaged on purpose to look like our noisy data** (explained below) | Yes, public | To *learn* how alive/dead and possession look in tracking data. About 2,000 possessions instead of 17. |
| **Check 1: "other matches"** | Each time, we keep one Bundesliga match out of the learning and test on it | Yes | Did the program learn the *concept*? (biased for our match, see below) |
| **Check 2: "public window"** | The 5 public minutes of our match, as they are | Yes | Only: is the program *correct* (right frames, right letters, right file)? |
| **Check 3: "public window, ball hidden"** | The same 5 minutes, with the ball artificially hidden using real gap patterns from the rest of our match | Yes | Does it survive the hard situation? |
| **Check 4 (optional): "hand-labelled minutes"** | ~12 extra minutes of our match where **two teammates watch the video and write the correct letters**, independently | Yes, made by us | An extra practice exam on *our* match. Never fed to the program, never submitted. |

**Why we damage the Bundesliga data.** Their tracking is clean; ours is not. If the program learned
on clean data, it would expect a perfect world. So before learning we hide their ball in long gaps of
the same lengths as ours, break player tracks into pieces, and add small position errors. Same idea as
practising a listening test with background noise because the real exam will be noisy.

**No part of our own match is used for learning.** Our match is only used for checking. The program
cannot memorise the 17 public possessions. One explicit exception, decided now: at most **two scalar
knobs** (how much to trust the evidence, how reluctant to switch) may be set on the public window,
and they are reported as such in the technical document. Everything else (rule thresholds, duration
statistics, learned model) comes from the Bundesliga matches or from football reasoning.

### Honest warning: every check is biased in some way

| Check | Big enough? | From our match? | Likely bias for the final score |
|---|---|---|---|
| 1. Other matches | yes (~2,000 possessions) | no | **optimistic**: different league, cleaner data, noise imitated by us |
| 2. Public window | no (17) | yes | **optimistic**: ball visible 88% instead of 59% |
| 3. Public window, ball hidden | no | yes, but artificial gaps | unknown direction |
| 4. Hand-labelled minutes | small (~40–60) | yes | closest to fair, but humans may label slightly differently from the official key |

**No check is both big and from our match.** So we never read one number and say "this is our
score". Section 8 explains how we decide anyway.

## 7. Our approach, step by step

The program has two parts: an **evidence reader** that looks at each frame, and a **storyteller**
that turns the per-frame evidence into a calm sequence of possessions.

```
tracking + tags ─► (1) clues per frame ─► (2) evidence: how likely H, A, D ─► (3) storyteller ─► letters
                                              ▲ version 1: our formulas
                                              ▲ version 2: small model learned on the 7 matches
```

### Step 1. Clues per frame (measurements, no learning)

From the tracking, for every frame we compute simple numbers ("clues"). Examples:

- **Ball clues** (when the ball is visible): how far is the closest Barça player? the closest
  Athletic player? is the ball outside the pitch lines? how fast is it moving?
- **Stoppage clues:** are most players walking or standing still? (After a foul or a ball out, the
  whole pitch slows down.) Is the ball resting at a corner or on the touchline?
- **Team-shape clues** (work without the ball): the team with the ball usually spreads out and
  pushes forward; the defending team closes up and steps back. Where are the teams moving to?
- **Tag clues:** the Smart Tagging says "corner kick here", "Athletic build-up there". Useful hints,
  but sometimes wrong, so they count as clues, not answers.
- **Data-quality clues:** how long has the ball been missing? how many players are visible?

The same clues can be computed on our match and on the Bundesliga matches, **except the tag clues**
(the Bundesliga data has no Smart Tagging). That matters for step 2.

### Step 2. Evidence: how likely is H, A or D at this frame?

We turn the clues into three numbers per frame, e.g. "70% Barça, 20% Athletic, 10% dead".

- **Version 1 (our formulas):** we write the rules ourselves. Example: "the closer the nearest Barça
  player is to the ball compared to the nearest Athletic player, the higher the Barça number". No
  learning. This is what the competitors already do. **Our first submission uses this.**
- **Version 2 (learned):** a small ML model learns, from the 7 damaged Bundesliga matches, how the
  clues relate to H/A/D. The model type is *gradient boosting*: many small yes/no rules ("are more
  than 15 players slower than 1 m/s?") combined into one answer; fast on a laptop, and it gives the
  same result every run. It can only learn from clues that exist in both datasets (tracking only);
  the tag clues are added afterwards as a separate hint.

**Will version 2 work? Honest answer: maybe (~40–50%).** It works if our damaged Bundesliga data
looks enough like our real data. It is most promising where behaviour is the same in any league
(players stopping at a foul, restart formations, team shape with and without the ball), which is
exactly where our formulas are weakest. It fails if our imitation of the noise is poor, or if
"dead ball" starts at a slightly different moment in their labels than in ours. So version 2 is
an upgrade that must beat version 1 on the checks; if it does not, we keep version 1.

### Step 3. The storyteller: a "stubborn" model that removes flicker

If we just took the highest number on every frame, we would get flicker, and flicker is fatal for
the score. So we use a **Hidden Markov Model (HMM)**. In plain words, it is a reasoning rule:

> "Possession does not change on a whim. I only switch from Barça to Athletic, or to dead ball, if
> the evidence for the switch is strong enough and lasts long enough. A single odd frame is noise."

It reads a **whole half at once** (we are not live, so it can look ahead) and chooses the single
most believable story, e.g. "Barça from 0:00 to 0:14, then Athletic until 0:22, then dead ball until
0:40...". The method that finds this best story is called the *Viterbi algorithm*.

When the ball is invisible, the storyteller **keeps the current owner** unless the team-shape and
stoppage clues clearly say something changed. Missing ball means "less evidence", not "dead ball"
(the organisers' example program makes exactly that mistake).

**Upgrade (optional):** an HMM has no sense of time ("how long do possessions usually last?"). A
variant called **HSMM** (hidden *semi*-Markov model) adds this, using durations learned from the
Bundesliga matches. It may keep short real possessions while still ignoring flicker. Kept only if it
wins on the checks.

### Step 4. Parallel track from week 1: can we find the ball in the video?

This does not depend on steps 1–3, so one person starts it in week 1.

- **The test (1.5 days):** take frames where the tracking *did* see the ball, pretend it did not,
  and check whether a ball detector finds it in the video. Count separately for the hard situations
  that really go missing: crowds and balls in the air.
- **Decision:** if the detector finds at least 1 in 4 balls in those hard situations, with very few
  false alarms, the ball step becomes a main track: we run it on the gap frames and give the found
  ball positions to step 1. If not, we stop and lose little.
- Expected gain if it works: a few points of score (guess: +0.01 to +0.07). Cost: laptop time, or a
  few euros of rented GPU. It also brings the "weights" question of section 9.

### Step 5. Produce, check and package the file

One command reads the data, runs steps 1–3, writes the 199,319-row file, and runs the organisers'
format checker. Running it twice must give the identical file.

Three structural rules are applied last, from the published constants in `games.py` (FAQ.md
"Which frames are actually graded?"); none of them is a guess:

- Frames outside the two graded blocks (before 11,185; 90,652–110,807; after 190,197) are `X`.
  They are never scored, so `X` is free there and avoids inventing possessions.
- Frames 110,808–113,741 (graded, before the second-half kick-off) are `D`, never `X`: the
  scorer back-fills a leading `X` with our first real label, which would stretch our first
  second-half possession by 2,934 frames and destroy its overlap.
- The storyteller restarts at each half; no possession carries across halftime.

## 8. How we will know we are improving, and what we cannot know

### How we decide between two versions

Every new version is run on all four checks (section 6) and compared with the current best. We look
at the **direction** of the change on every check, not at one number:

- **Check 1 (other matches)** is the main judge for "did it learn better", because it has ~2,000
  possessions. With 17 possessions the uncertainty is about ±0.24; you need ~385 possessions for
  about ±0.05.
- **Check 2 (public window)** must stay correct; small differences there are ignored.
- **Check 3 (ball hidden)** must not collapse.
- **Check 4 (hand-labelled minutes)**, if we do it, must not get worse.
- **Shape check on the full match** (no answers needed): believable number and lengths of
  possessions, no endless or tiny ones.

**Rule, decided now before seeing any result:** a new version replaces the current one only if it is
clearly better (≥ 0.04 points), wins on at least 5 of the 7 Bundesliga matches, and does not get worse
on our match's checks. Otherwise we keep the simpler one.

### What we cannot know before the deadline

- **Our real score.** No check is both large and from our match, and the large one is optimistic. We
  will probably score **below** what our checks show. The ranges in section 11 are educated guesses.
- **Whether a version that looks better is truly better.** Agreement across all checks makes it
  likely, not certain. The organisers showed one example where the ranking flips.

## 9. What we hand in, and the rules about it

### The three deliverables

1. **The CSV file** with our 199,319 letters. This is the only thing that is scored.
2. **A public code repository** (our code on GitHub, visible to anyone). It must contain everything
   needed so that **anyone who has the match data can run one command and get exactly the same CSV**,
   identical to the last character. The organisers may re-run the winners' code to check this.
3. **A short technical document** (a few pages): what we built and why, what data we used and
   dropped, our possession rules, limitations, how to re-run.

So we do **not** upload "a model" for them to use. We publish **the full recipe**: data in, CSV out.

### Explained like you are five: the cake contest

Imagine a cake contest.

- **The match data is a box of special ingredients** that the organisers lent to every team. The box
  belongs to them. You promised not to give it to anyone else.
- **Our CSV is the cake.** The judges taste the cake. That is our score.
- **Our code is the recipe.** We must pin the recipe on a public wall so the judges can bake the same
  cake again with their own box of ingredients and check that we did not cheat. Everyone in the world
  can read that wall.
- **Rule:** you may pin the *recipe* on the wall. You may **not** pin the *ingredients* on the wall,
  and you may not pin anything from which a stranger could rebuild the ingredients (for example,
  photocopies of every page of the box's contents). That is what "do not publish derived data that
  could reconstruct it" means: *derived data* = things made from the data, like a table with the
  position of every player in every frame, or clips cut from the video. If it lets someone get the
  data without signing the organisers' agreement, it must not go on the public wall.

Concretely, what may and may not go into our public repository:

| Thing | In the public repo? | Why |
|---|---|---|
| Our code (recipe) | **yes** | required |
| Match video, tracking, tags (ingredients) | **no** | not ours to give away |
| Tables made from the match: positions per frame, our per-frame clues, video clips | **no** | someone could rebuild the data from them |
| Summary numbers (a score, "the ball is missing 41% of the time") | **yes** | cannot rebuild anything |
| Our final CSV | not in the repo (it is uploaded through the form) | it is the answer file |
| Public Bundesliga data | **no copy; a script that downloads it** | it is someone else's; the script lets judges get it themselves |

### And what are "weights"?

When a program *learns* (version 2 in step 2, or a ball detector), the result of learning is a file
of numbers, called **weights**. Cake analogy: it is like a **secret sauce** you prepared beforehand.

The judges must be able to bake the identical cake. There are two ways:

- **Give them the sauce** (publish the weights file). Fast and always identical. But: if the sauce
  was made *from the special ingredients* (a detector that learned from the match video), is the sauce
  "derived data"? A file of weights almost certainly cannot rebuild a video, but it is a grey area.
  That is what one team asked the organisers in issue #2.
- **Give them the sauce recipe** (the training code), so they make the sauce themselves. Then the
  sauce must come out **exactly** the same every time, or the final CSV changes and fails the check.
  Small models on a normal computer can be made exactly repeatable. Big models trained on a graphics
  card (GPU) often come out slightly different each time, and the judges might not even have a GPU.
  That is the second question in issue #2.

What it means for us:

| Part of our system | Weights problem? |
|---|---|
| Version 1 (our formulas) + storyteller | **none**: nothing learned, nothing to ship |
| Version 2 (gradient boosting on Bundesliga) | **small**: it learns in minutes on a normal computer and can repeat exactly, so we ship the training code plus the download script. The sauce is made from public ingredients, not from the match. |
| Video ball detector (step 4) | **real**: learned from the match video, probably needs a GPU. We would follow the answer to issue #2 if one appears; otherwise keep it optional so our submission never depends on it. Our own repo rule "no weights in Git" would also have to change. |

## 10. Plan, team split, budget

### Work packages

| ID | Work | Done when |
|---|---|---|
| W0 | Download data; small scripts print summary facts (gap lengths, does the ball vanish when dead, etc.); reproduce the organisers' example | Example scores reproduce exactly |
| W1 | Steps 1–3 with version 1 (our formulas + HMM) → **first real submission** | Beats the example program on checks 2 and 3 |
| W2 | More clues: stoppage, team shape, tags | Each clue kept only if it helps on the checks |
| W3 | Bundesliga data: download, damage it to look like ours, learn version 2 | Beats version 1 by the rule in section 8 |
| W4 | HSMM upgrade | Beats the HMM by the rule in section 8 |
| W5 | The checking tool: one command, all checks | Used for every decision |
| W6 | (optional) Hand-labelling of ~12 minutes by two people | The two people agree with each other, and with the public 5 minutes |
| W7a | **Broadcast tracking as a second ball source (parallel, week 1)**: align the two clocks by player formations, measure how many tactical gap frames the broadcast ball covers (section 5) | Aggregate coverage number printed; go if it covers a meaningful share of gap frames (threshold set before running: ≥ 15%) |
| W7 | Video ball test (parallel, week 1–2, only if W7a is not enough), then ball recovery if it passes | Decision rule of section 7, step 4 |
| W8 | Reproducibility + technical document | Same file twice; checker OK |

W0 also includes a manual step: the Bundesliga files are behind a browser-only download (the
hosting site blocks scripted downloads), so one person downloads them by hand into the private work
folder. For the judges this means our repository must either ship the small learned model (made
from public data only, with a checksum and the training script) or document the manual download;
see the decision in section 9.

### Timeline (deadline 17 November)

| Week | Dates | Main line | Parallel line |
|---|---|---|---|
| 0 | 22–26 Oct | Data, facts, checking-tool skeleton | — |
| 1 | 27 Oct–2 Nov | **First submission uploaded** (insurance; the form can be edited) | W7a broadcast-ball coverage (2 days) → go / no-go; then video ball test (1.5 days) if needed |
| 2 | 3–9 Nov | Version 2 on Bundesliga, HSMM, more clues | Ball recovery from whichever source passed, only if "go" |
| 3 | 10–14 Nov | Final choice using the rule in section 8; technical document | Merge ball recovery if it passed the checks |
| — | 15–17 Nov | Buffer, final check, upload | |

### Team split (no football or ML background needed for several roles)

**Rule check:** RULES.md and OVERVIEW.md say a team has **3 to 5 members**. The roles below were
written for six people; if we are six, one person must stay off the registered team or the roles
must be merged (the natural merge is "Checks" into "Lead", or "Video ball" into "Football
clues"). Confirm the registered head-count before assigning roles.

| Role | Work | Who fits |
|---|---|---|
| Lead / integrator | plan, main command, final decision, W8 | strongest programmer |
| Storyteller | W1, W4 | likes logic and probabilities |
| Football clues | W2, checks that rules make football sense | the football fan |
| Learning from Bundesliga | W3 | most ML experience |
| Checks | W5, W6 coordination, shape checks | careful and methodical |
| Video ball | W7, then technical document | curious, likes images |

### Budget

The main line runs on a laptop: the storyteller in seconds, version 2 in minutes. **Expected cloud
cost: €0.** The video track may need a few hours of a rented GPU (≈ €2–5) if the laptop is too slow;
see the open question in section 11. Cap: €15.

### Data rule (applies to everyone)

The match data may not be uploaded to any online AI (ChatGPT, Claude, Gemini, ...). Programs read
the data on our machines and print only summaries (scores, counts). See `docs/DATA_POLICY.md`.

## 11. Chances, risks, open questions

### Chances (subjective estimates)

| Outcome | Full-match score | Chance |
|---|---|---|
| Beat the example program | > 0.25 | ~90% |
| Most likely result | 0.30–0.45 | — |
| If version 2 and/or ball recovery work | 0.40–0.55 | ~40–50% |
| Excellent | > 0.55 | ~10–20% |

Other teams report 0.50–0.61 on the public 5 minutes, but those minutes are the easy part; on the
full match everyone will score lower.

### Main risks

| Risk | What we do |
|---|---|
| Choosing versions by the 5 public minutes | Decision rule of section 8 |
| Bundesliga "dead ball" starts at a slightly different moment than in our answer key | Measure the timing difference on the public 5 minutes and correct it once |
| Our damaged Bundesliga data does not look like our real data | Compare summary statistics of both; adjust the damage |
| The video detector does not find the hidden balls | Test first (1.5 days); stop if it fails |
| Too ambitious for a part-time team | First submission in week 1; every extra is optional |
| Program cannot regenerate the identical file | Fixed random seeds, fixed versions, double run before submitting |

### Open questions (we do not contact the organisers for now; we design around them)

1. How many hand-labelled minutes count as "small" for checking (issue #8, unanswered)? Check 4
   stays optional and small.
2. May weights learned from the match video be published, and may re-running need a GPU (issue #2,
   unanswered)? The video track stays optional, so our submission never depends on it.
3. May the match video be processed on a rented cloud GPU under the data terms? Default: process it
   on our own laptops.

### Decisions needed from the team

1. Approve installing `scikit-learn` (the library for version 2).
2. Hand-labelled minutes (W6): yes or no.
3. Confirm the registered team size (3–5 by the rules) and assign the roles.
4. ~~Download the data~~ Done 2026-10-22; all nine assets present.
5. Allow a small model file learned from **public** data only (Bundesliga) in the repository, with
   a checksum and the training script? Our current rule says "no weights in Git"; it was written
   for weights learned from the match. Without this, judges need the manual Bundesliga download to
   reproduce our file byte-for-byte.

## 12. Glossary

| Term | Meaning |
|---|---|
| frame | one still image of the video; 25 per second |
| ground truth / answer key | the correct letter for each frame |
| tracking data | positions and speeds of players and ball, per frame, computed from video |
| possession | uninterrupted period of one team having the alive ball |
| dead ball (`D`) | game stopped by the referee |
| flicker | a wrong letter for one or a few frames inside a possession |
| spellF1 | the ranking score: share of real possessions found, minus invented ones |
| macroF1 | the tie-break score: share of correct frames per letter |
| clue / feature | a number computed from the data that helps decide (e.g. distance ball–player) |
| label | the correct answer attached to an example |
| model | a program whose behaviour is learned from examples |
| training set | examples with answers the model learns from |
| validation set / check | examples with answers used to compare versions; not learned from |
| test set | the data of the final grade; answers hidden |
| weights | the file of numbers a model ends up with after learning ("the secret sauce") |
| derived data | anything made from the match data (tables, clips); may not be published if it could rebuild the data |
| reproducible | anyone running our code on the data gets the identical CSV |
| HMM | the "stubborn storyteller": turns per-frame evidence into a calm sequence |
| HSMM | an HMM that also knows typical durations |
| gradient boosting | a fast learning method combining many small yes/no rules |
| GPU | graphics card; makes training image models much faster |
| Smart Tagging | automatic notes on match phases and set pieces |
| set piece / restart | the way play resumes after a dead ball (corner, throw-in, free kick...) |
| IDSSE / Sportec Open DFL | free dataset of 7 German professional matches with tracking and per-frame answers |
