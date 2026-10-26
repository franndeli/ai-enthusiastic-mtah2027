"""Aggregate-only profile of frame,label CSVs: how many possessions and stoppages, and how long. Use
it to compare a submission with the public ground truth's conventions (CHALLENGE.md: "use it to
understand what is expected"), never to tune on it. Prints counts only: no rows, no frame numbers.

usage: python scripts/label_profile.py <csv> [<csv> ...] [--window public|graded]

--window public (default) profiles only the public 5 minutes, video frames 11185 ... 18684
(DATA.md), so a full-match submission and the public ground truth are compared on the same frames;
--window graded profiles every non-X frame of each file.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import LABELS, A, D, H, X  # noqa: E402

PUBLIC = (11185, 18685)
EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, np.inf)
NAMES = ("< 2 s", "2-5 s", "5-10 s", "10-20 s", ">= 20 s")


def runs(lab):
    cut = np.flatnonzero(np.diff(lab) != 0) + 1
    return [(int(s), int(e), int(lab[s])) for s, e in zip(np.r_[0, cut], np.r_[cut, len(lab)])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--window", default="public", choices=("public", "graded"))
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()
    print("%-34s %6s %6s %7s  %s  | %6s %7s" % (
        "file (" + args.window + ")", "frames", "poss", "mean s", "  ".join("%7s" % n for n in NAMES),
        "D runs", "D mean"))
    for path in args.csv:
        lab = pd.read_csv(path, dtype={"label": "object"})["label"].str.strip()
        lab = lab.map({c: i for i, c in enumerate(LABELS)}).to_numpy(dtype=np.int8)
        if args.window == "public":
            lab = lab[PUBLIC[0]:PUBLIC[1]]
        lab = lab[lab != X]
        rr = runs(lab) if len(lab) else []
        poss = [(e - s) / args.fps for s, e, t in rr if t in (H, A)]
        dead = [(e - s) / args.fps for s, e, t in rr if t == D]
        hist = np.histogram(poss, bins=EDGES)[0] if poss else np.zeros(5, dtype=int)
        print("%-34s %6d %6d %7.1f  %s  | %6d %7.1f" % (
            os.path.basename(path)[:34], len(lab), len(poss), np.mean(poss) if poss else 0.0,
            "  ".join("%7d" % v for v in hist), len(dead), np.mean(dead) if dead else 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
