"""Aggregate-only report on the missing ball for the match: how much of it is missing and in which
gaps, and what each given submission labels while the ball is missing. Prints aggregates only: no
rows, no frame numbers, no positions.

usage: python scripts/gap_report.py [--data DIR] [--game BAR-ATH] [submission.csv ...]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import LABELS  # noqa: E402
from possession.gaps import BUCKETS, buckets, gap_summary, label_profile  # noqa: E402
from possession.pipeline import PRIVATE_ROOT, game_constants  # noqa: E402
from possession.tracking import read_epts  # noqa: E402

# DATA.md, "The public ground truth": the public 5 minutes are video frames 11185 ... 18684.
PUBLIC_WINDOW = {"BAR-ATH": (11185, 18685)}


def read_labels(path: str, n: int) -> np.ndarray:
    lab = pd.read_csv(path, dtype={"label": "object"})["label"].str.strip()
    codes = lab.map({c: i for i, c in enumerate(LABELS)})
    if len(codes) != n or codes.isna().any():
        raise SystemExit("%s: expected %d rows of H/A/D/X" % (os.path.basename(path), n))
    return codes.to_numpy(dtype=np.int8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="*", help="frame,label CSVs to profile")
    parser.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR",
                                                         os.path.join(PRIVATE_ROOT, "data")))
    parser.add_argument("--game", default="BAR-ATH")
    args = parser.parse_args()

    games, n_frames, fps, pitch, blocks = game_constants(args.game)
    paths = games.asset_paths(args.game, data_dir=args.data)
    if not all(os.path.isfile(paths[k]) for k in ("atd_meta", "atd_raw")):
        print("match data not found; set MTAH_DATA_DIR to the folder that contains tactical/")
        return 2
    trk = read_epts(paths["atd_meta"], paths["atd_raw"], pitch=pitch, n_frames=n_frames)
    bucket = buckets(trk.ball[:, 0], blocks, fps)

    s = gap_summary(bucket)
    print("ball missing on %.1f%% of %d graded frames (DATA.md: 41%%)"
          % (100 * s["missing_share"], s["graded_frames"]))
    print("  missing frames by gap length: < 1 s %.1f%% (%d gaps), 1-10 s %.1f%% (%d gaps), "
          ">= 10 s %.1f%% (%d gaps)" % (100 * s["missing_in_1"], s["gaps_1"],
                                       100 * s["missing_in_2"], s["gaps_2"],
                                       100 * s["missing_in_3"], s["gaps_3"]))
    window = PUBLIC_WINDOW.get(args.game)
    if window:
        w = gap_summary(bucket, window)
        print("  public 5 minutes: ball missing on %.1f%% of frames, %.1f%% of those in gaps >= 10 s"
              % (100 * w["missing_share"], 100 * w["missing_in_3"]))

    for path in args.submissions:
        prof = label_profile(bucket, read_labels(path, trk.n))
        print("\n%s" % os.path.basename(path))
        print("  %-12s %8s %6s %6s %6s %6s %14s" % ("", "frames", "H", "A", "D", "X",
                                                 "team changes"))
        for name in BUCKETS:
            r = prof[name]
            changes = ("%d in %d gaps" % (r["team_changes"], r["gaps"])) if "gaps" in r else ""
            print("  %-12s %8d %5.1f%% %5.1f%% %5.1f%% %5.1f%% %14s" % (
                name, r["frames"], 100 * r["H"], 100 * r["A"], 100 * r["D"], 100 * r["X"],
                changes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
