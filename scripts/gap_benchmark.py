"""Where the missing ball costs us, on the external benchmark (Metrica sample games, leave one game
out), where the truth is known: per gap-length bucket, the label mix of truth / v3 / v4, frame
accuracy and possession changes inside gaps; and which true possessions are missed, by length and
cause. Other games only; prints aggregates only.

usage: python scripts/gap_benchmark.py <sample-data/data> [--noise default,more_missing_in_play]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import H, A, D, X  # noqa: E402
from possession import features as F  # noqa: E402
from possession.decode import segment  # noqa: E402
from possession.external import degrade, load_sample_game, random_noise  # noqa: E402
from possession.gaps import BUCKETS, buckets  # noqa: E402
from possession.model import Model  # noqa: E402

VARIANTS = {"v3": (F.TEAM_FEATURES_V3, F.DEAD_FEATURES, "default"),
            "v4": (F.TEAM_FEATURES_V4, F.DEAD_FEATURES_V4, "random")}
NOISE = {"default": {}, "more_missing_in_play": {"p_miss_alive": 0.35, "p_miss_dead": 0.55}}
TRAIN_SEEDS, TEST_SEEDS = (0, 1), (10, 11, 12)


def gap_runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


LENGTHS = (0.0, 2.0, 5.0, 10.0, 20.0, np.inf)
LENGTH_NAMES = ("< 2 s", "2-5 s", "5-10 s", "10-20 s", ">= 20 s")
CAUSES = ("merged", "wrong team", "dead", "fragmented")


def possessions(lab, blocks):
    """Maximal H or A runs inside the blocks: [(start, stop, team)]."""
    out = []
    for a, b in blocks:
        seg = lab[a:b]
        cut = np.flatnonzero(np.diff(seg) != 0) + 1
        for s, e in zip(np.r_[0, cut], np.r_[cut, len(seg)]):
            if seg[s] in (H, A):
                out.append((a + int(s), a + int(e), int(seg[s])))
    return out


def missed_by_length(gt, pred, blocks, fps):
    """Per true-possession length: count, matched (same team, IoU >= 0.5, one-to-one, greedy --
    an approximation of the official spell matching) and, for the missed ones, why."""
    g_runs, p_runs = possessions(gt, blocks), possessions(pred, blocks)
    cand = []
    for gi, (a, b, t) in enumerate(g_runs):
        for pi, (c, d, u) in enumerate(p_runs):
            if u == t and c < b and d > a:
                iou = (min(b, d) - max(a, c)) / (max(b, d) - min(a, c))
                if iou >= 0.5:
                    cand.append((iou, gi, pi))
    matched, used = set(), set()
    for _, gi, pi in sorted(cand, reverse=True):
        if gi not in matched and pi not in used:
            matched.add(gi)
            used.add(pi)
    table = np.zeros((len(LENGTH_NAMES), 2 + len(CAUSES)))
    for gi, (a, b, t) in enumerate(g_runs):
        k = int(np.searchsorted(LENGTHS, (b - a) / fps, side="right")) - 1
        table[k, 0] += 1
        if gi in matched:
            table[k, 1] += 1
            continue
        seg = pred[a:b]
        if np.mean(seg == D) >= 0.5:
            cause = "dead"
        elif np.mean(seg == t) < 0.5:
            cause = "wrong team"
        else:
            inside = [r for r in p_runs if r[2] == t and r[0] < b and r[1] > a]
            best = max(inside, key=lambda r: min(r[1], b) - max(r[0], a))
            cause = "merged" if best[1] - best[0] > b - a else "fragmented"
        table[k, 2 + CAUSES.index(cause)] += 1
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", help="folder with Sample_Game_1..3 (metrica-sports/sample-data/data)")
    parser.add_argument("--noise", default="default,more_missing_in_play")
    args = parser.parse_args()
    t0 = time.time()
    games = {g: load_sample_game(g, args.root) for g in (1, 2, 3)}
    models = {}
    for name, (team, dead, mode) in VARIANTS.items():
        train = {}
        for g, (trk, gt) in games.items():
            for s in TRAIN_SEEDS:
                deg = degrade(trk, gt, noise=random_noise(s) if mode == "random" else None, seed=s)
                train[(g, s)] = (gt, F.compute(deg))
        for test in games:
            keys = [k for k in train if k[0] != test]
            models[(name, test)] = Model.fit([train[k][1] for k in keys],
                                             [train[k][0] for k in keys],
                                             team_names=team, dead_names=dead)
        del train
    names = ["truth"] + list(VARIANTS)
    for sc in args.noise.split(","):
        mix = {(n, k): np.zeros(3) for n in names for k in range(4)}
        hits = {(n, k): 0 for n in VARIANTS for k in range(4)}
        changes = {(n, k): [0, 0] for n in names for k in (1, 2, 3)}
        frames = np.zeros(4)
        missed = {n: np.zeros((len(LENGTH_NAMES), 2 + len(CAUSES))) for n in VARIANTS}
        copies = 0
        for test, (trk, gt) in games.items():
            for s in TEST_SEEDS:
                deg = degrade(trk, gt, noise=NOISE[sc], seed=s)
                f = F.compute(deg)
                b = buckets(deg.ball[:, 0], deg.blocks, deg.fps)
                b = np.where(gt != X, b, -1)
                labs = {"truth": gt}
                for n in VARIANTS:
                    m = models[(n, test)]
                    labs[n] = segment(m.dead_logit(f), m.team_logit(f), deg.blocks, deg.n,
                                      cost_dead=40.0, cost_team=8.0)
                    missed[n] += missed_by_length(gt, labs[n], deg.blocks, deg.fps)
                copies += 1
                for k in range(4):
                    m = b == k
                    frames[k] += m.sum()
                    for n, lab in labs.items():
                        mix[(n, k)] += [np.sum(lab[m] == c) for c in (H, A, D)]
                        if n != "truth":
                            hits[(n, k)] += int(np.sum(lab[m] == gt[m]))
                        if k > 0:
                            for a, e in gap_runs(m):
                                team = lab[a:e][np.isin(lab[a:e], (H, A))]
                                changes[(n, k)][0] += int(np.sum(team[1:] != team[:-1]))
                                changes[(n, k)][1] += 1
        miss = frames[1:].sum()
        print("\ntest noise %s: ball missing on %.1f%% of graded frames; of the missing, "
              "< 1 s %.1f%%, 1-10 s %.1f%%, >= 10 s %.1f%%"
              % ((sc, 100 * miss / frames.sum()) + tuple(100 * frames[1:] / miss)))
        print("  %-12s %6s  %s  %s  %s" % (
            "bucket", "share", "  ".join("%-14s" % ("%s H/A/D" % n) for n in names),
            "  ".join("%s acc" % n for n in VARIANTS), "team changes per gap " + "/".join(names)))
        for k in range(4):
            cells = []
            for n in names:
                v = 100 * mix[(n, k)] / max(mix[(n, k)].sum(), 1)
                cells.append("%-14s" % ("%2.0f/%2.0f/%2.0f%%" % tuple(v)))
            acc = "  ".join("%5.1f%%" % (100 * hits[(n, k)] / max(frames[k], 1)) for n in VARIANTS)
            ch = "" if k == 0 else " / ".join("%.2f" % (changes[(n, k)][0] / max(changes[(n, k)][1], 1))
                                              for n in names)
            print("  %-12s %5.1f%%  %s  %s  %s" % (BUCKETS[k], 100 * frames[k] / frames.sum(),
                                                   "  ".join(cells), acc, ch))
        for n in VARIANTS:
            t = missed[n] / copies
            print("  %s: true possessions per game %.0f, recall by length (IoU >= 0.5) and why the "
                  "missed ones are missed, per game" % (n, t[:, 0].sum()))
            print("    %-9s %6s %7s   %s" % ("length", "true", "recall", "  ".join(
                "%-10s" % c for c in CAUSES)))
            for k, name in enumerate(LENGTH_NAMES):
                print("    %-9s %6.1f %6.1f%%   %s" % (name, t[k, 0], 100 * t[k, 1] / max(t[k, 0], 1e-9),
                                                     "  ".join("%-10.1f" % v for v in t[k, 2:])))
    print("total %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
