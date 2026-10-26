"""Download the Sportec Open DFL / IDSSE matches (public, CC-BY 4.0; other games, not the challenge
match) with kloppy and save each as a compact .npz (positions in cm, per-frame H/A/D labels from
BallPossession and BallStatus). About 2.3 GB is downloaded; the outputs are a few MB per match.

usage: pip install kloppy==3.19.1      (in a separate venv, with numpy==2.4.6 pandas==3.0.6)
       python scripts/fetch_dfl.py <out dir> [--matches J03WMX,J03WN1,...]
       python scripts/fetch_dfl.py <out dir> --meta meta.xml --raw positions.xml --name test

kloppy >= 3.19 downloads from the Hugging Face mirror (huggingface.co/datasets/pysport/idsse-data);
the figshare links used by kloppy 3.18 return an empty file to scripts. Manual alternative: download
the *matchinformation* and *positions_raw* XML files from doi.org/10.6084/m9.figshare.28196177 and
use --meta/--raw.

Prints, per match, only frame counts and label shares (these are public data).
"""
import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import A, D, H  # noqa: E402
from possession.dfl import MATCHES, load, save, to_arrays  # noqa: E402


def report(name: str, path: str, t0: float) -> None:
    trk, lab = load(path)
    graded = lab != 3
    print("  %s: %d frames in %d periods, %d player slots, alive H %.1f%% / A %.1f%% / D %.1f%%, "
          "%.1f MB, %.0fs" % (name, trk.n, len(trk.blocks), len(trk.team),
                             100 * np.mean(lab[graded] == H), 100 * np.mean(lab[graded] == A),
                             100 * np.mean(lab[graded] == D), os.path.getsize(path) / 1e6,
                             time.time() - t0), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", help="output folder for dfl_<match>.npz")
    parser.add_argument("--matches", default=",".join(MATCHES))
    parser.add_argument("--meta", help="local metadata XML instead of downloading")
    parser.add_argument("--raw", help="local positions XML instead of downloading")
    parser.add_argument("--name", default="local")
    args = parser.parse_args()
    try:
        from kloppy import sportec
    except ImportError:
        print("kloppy is not installed in this Python: run it from a separate venv with "
              "`pip install kloppy==3.19.1 numpy==2.4.6 pandas==3.0.6` (see the usage above)")
        return 2
    os.makedirs(args.out, exist_ok=True)
    if args.meta and args.raw:
        t0 = time.time()
        ds = sportec.load_tracking(meta_data=args.meta, raw_data=args.raw, only_alive=False,
                                   coordinates="sportec")
        path = os.path.join(args.out, "dfl_%s.npz" % args.name)
        save(to_arrays(ds), path)
        report(args.name, path, t0)
        return 0
    for match in [m.strip() for m in args.matches.split(",") if m.strip()]:
        path = os.path.join(args.out, "dfl_%s.npz" % match)
        if os.path.isfile(path):
            print("  %s: already there" % match)
            continue
        t0 = time.time()
        print("downloading and converting %s (a few minutes) ..." % match, flush=True)
        ds = sportec.load_open_tracking_data(match_id=match, only_alive=False,
                                             coordinates="sportec")
        save(to_arrays(ds), path)
        del ds
        gc.collect()
        report(match, path, t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
