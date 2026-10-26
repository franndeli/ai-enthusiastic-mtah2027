"""Public-GT forensics (ROUTE_A_PLAN.md T1): the public 5-minute ground truth's structure and how
the official spellF1/macroF1 react to deterministic perturbations of it, scored with the repo's own
scorer. Reads only public files (the public GT and the committed baselines). Run:
    scripts/py -m possession.public_gt_report
"""

import glob
import os
import sys

import numpy as np

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
sys.path.insert(0, EVAL_DIR)
import possession_metrics as pm  # noqa: E402  (imported the way run_evaluation.py does)
from run_evaluation import RESULTS_DIR, find_gt  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
FPS = pm.FPS
SEEDS = range(10)


# -- perturbations: pure functions over (codes, runs); frames outside the runs are left alone ----

def lag(codes, runs, k):
    """Every label k frames late inside each run (the run head repeats its first label)."""
    out = codes.copy()
    for start, stop in runs:
        out[start + k:stop] = codes[start:stop - k]
        out[start:start + k] = codes[start]
    return out


def inject_flicker(codes, runs, n, length, label, rng):
    """n spurious `length`-frame runs of `label` ('other' = opposing team) inside random spells."""
    out = codes.copy()
    spells = [s for s in pm.possession_spells(codes, runs) if s[1] - s[0] > length + 2]
    for i in rng.choice(len(spells), size=n, replace=n > len(spells)):
        start, stop, team = spells[i]
        at = int(rng.integers(start + 1, stop - length))
        out[at:at + length] = (A if team == H else H) if label == "other" else label
    return out


def majority_smooth(codes, runs, window):
    """Centred 3-class majority over +/- window frames (ties: H, then A, then D)."""
    out = codes.copy()
    for start, stop in runs:
        seg = codes[start:stop]
        cum = np.vstack([np.concatenate(([0], np.cumsum(seg == c))) for c in (H, A, D)])
        idx = np.arange(len(seg))
        low, high = np.maximum(0, idx - window), np.minimum(len(seg), idx + window + 1)
        out[start:stop] = np.array([H, A, D])[np.argmax(cum[:, high] - cum[:, low], axis=0)]
    return out


def mask(codes, runs, pick, value):
    """Copy with `value` on the frames that pick(segment_list) returns as (start, stop) pairs."""
    out = codes.copy()
    for start, stop in pick(pm.segments(codes, runs)):
        out[start:stop] = value
    return out


def boundaries(segs):
    return [(s[0], s[1], s[2], prev[2]) for prev, s in zip(segs, segs[1:]) if prev[1] == s[0]]


def x_middle(segs):
    return [(a + (b - a) // 4, b - (b - a) // 4) for a, b, k in segs if k in (H, A)]


def x_boundary(segs, half=FPS):
    return [(max(segs[0][0], a - half), a + half) for a, _, _, _ in boundaries(segs)]


def d_stretches(segs):
    return [(a, b) for a, b, k in segs if k == D]


def no_d(codes, runs):
    """Every D stretch relabelled with the previous team (the next one at the head of a run)."""
    return pm.fill_abstentions(mask(codes, runs, d_stretches, X), runs)


def shift_d_edges(codes, runs, k):
    """D stretches grown (k > 0) or shrunk (k < 0) by |k| frames at both ends."""
    out = codes.copy()
    segs = pm.segments(codes, runs)
    for i, (a, b, lab) in enumerate(segs):
        if lab != D:
            continue
        if k > 0:
            out[max(a - k, segs[0][0]):a] = D
            out[b:min(b + k, segs[-1][1])] = D
        else:
            m = min(-k, (b - a) // 2)
            if i > 0:
                out[a:a + m] = segs[i - 1][2]
            if i + 1 < len(segs):
                out[b - m:b] = segs[i + 1][2]
    return out


def merge_short(codes, runs, min_frames, kinds):
    """Segments of label in `kinds` shorter than min_frames take the previous segment's label."""
    out = codes.copy()
    segs = pm.segments(codes, runs)
    for i, (a, b, lab) in enumerate(segs):
        if lab in kinds and b - a < min_frames:
            out[a:b] = out[segs[i - 1][0]] if i > 0 else segs[i + 1][2]
    return out


# -- reporting ---------------------------------------------------------------------------------

def score(gt, pred):
    r = pm.evaluate(gt, pred)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"]


def row(name, gt, pred):
    spell, macro, n = score(gt, pred)
    print("  {:<44} spellF1 {:.4f}  macroF1 {:.4f}  spells {:>3}".format(name, spell, macro, n))
    return spell


def seeded_mean(gt, make):
    """Mean (spellF1, macroF1, spells) over SEEDS draws of a random perturbation."""
    scores = np.array([score(gt, make(np.random.default_rng(seed))) for seed in SEEDS])
    return scores.mean(axis=0)


def print_structure(gt, runs):
    segs = pm.segments(gt, runs)
    spells = pm.possession_spells(gt, runs)
    dur = lambda items: np.array([b - a for a, b, *_ in items]) / FPS  # noqa: E731
    print("graded spans (frames, half-open):", runs)
    print("segments {}  spells {}  D stretches {}".format(
        len(segs), len(spells), sum(k == D for _, _, k in segs)))
    print("\n  #  label  start   frames   seconds")
    for i, (a, b, k) in enumerate(segs):
        print("  {:>2}  {}  {:>7}  {:>6}  {:>7.2f}".format(i, pm.LABELS[k], a, b - a, (b - a) / FPS))
    print("\nsequence:", " ".join(pm.LABELS[k] for _, _, k in segs))
    for name, items in (("H", [s for s in spells if s[2] == H]),
                        ("A", [s for s in spells if s[2] == A]),
                        ("all spells", spells),
                        ("D", [s for s in segs if s[2] == D])):
        d = dur(items)
        print("  {:<10} n={:>2}  min {:6.2f}s  median {:6.2f}s  mean {:6.2f}s  max {:6.2f}s".format(
            name, len(d), d.min(), np.median(d), d.mean(), d.max()))
    labs = [k for _, _, k in segs]
    direct = sum(p != q and D not in (p, q) for p, q in zip(labs, labs[1:]))
    via_d = [(p, q) for p, d, q in zip(labs, labs[1:], labs[2:]) if d == D and D not in (p, q)]
    print("hand-overs: direct H<->A {}  |  via D: team change {}, same team resumes {}".format(
        direct, sum(p != q for p, q in via_d), sum(p == q for p, q in via_d)))


def main():
    gt = pm.label_codes(pm.read_label_csv(find_gt("public")))
    runs = pm.valid_runs(gt)

    print("== a. public GT structure ==")
    print_structure(gt, runs)
    assert score(gt, gt)[0] == 1.0

    print("\n== b(i). uniform lag ==")
    for k in (1, 3, 8, 12, 25, 50):
        spell = row("lag {} frames ({:.2f} s)".format(k, k / FPS), gt, lag(gt, runs, k))
        if k == 1:
            assert spell == 1.0, "EVALUATION.md: a one-frame offset is free"

    print("\n== b(ii). flicker injection (mean over {} seeds) ==".format(len(SEEDS)))
    for label in ("other", D):
        for length in (1, 5, 12, 25):
            for n in (1, 3, 5, 10, 17):
                spell, macro, n_sp = seeded_mean(
                    gt, lambda rng: inject_flicker(gt, runs, n, length, label, rng))
                print("  {:<5} len {:>2}  n {:>2}  spellF1 {:.4f}  macroF1 {:.4f}  spells {:5.1f}".format(
                    "team" if label == "other" else "D", length, n, spell, macro, n_sp))

    print("\n== b(iii). centred majority smoothing of the GT ==")
    for window in (25, 50, 125):
        row("majority +/-{} frames".format(window), gt, majority_smooth(gt, runs, window))

    print("\n== b(iv). X policies ==")
    row("X middle 50% of each possession", gt, mask(gt, runs, x_middle, X))
    row("X +/-1 s around each boundary", gt, mask(gt, runs, x_boundary, X))
    row("X over every D stretch", gt, mask(gt, runs, d_stretches, X))

    print("\n== b(v). D errors ==")
    row("no D at all (D -> previous team)", gt, no_d(gt, runs))
    for k in (3, 12, 25, 50):
        row("D stretches grown {} frames each end".format(k), gt, shift_d_edges(gt, runs, k))
        row("D stretches shrunk {} frames each end".format(k), gt, shift_d_edges(gt, runs, -k))

    print("\n== b(vi). merge short segments into the previous one ==")
    for t in (1, 2, 3, 5, 8, 10):
        row("spells < {} s merged".format(t), gt, merge_short(gt, runs, t * FPS, (H, A)))
        row("any segment (H/A/D) < {} s merged".format(t), gt, merge_short(gt, runs, t * FPS, (H, A, D)))

    print("\n== c. committed baselines vs public GT ==")
    expected = {"baseline_raw": 0.0378, "baseline_smooth25": 0.2868,
                "baseline_smooth50": 0.3256, "baseline_smooth125": 0.4242}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "baseline_*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        spell = row(name, gt, pm.label_codes(pm.read_label_csv(path)))
        assert round(spell, 4) == expected[name], name


if __name__ == "__main__":
    main()
