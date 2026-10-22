#!/usr/bin/env python3
"""Delete experiment CSVs and reports from evaluation/ so the board shows only what matters.

Keeps the upstream baselines and any names passed as arguments. Prints file names only.

    scripts/py scripts/clean_results.py ai-enthusiastic
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
REPORTS = ROOT / "evaluation" / "reports"


def main() -> int:
    keep = set(sys.argv[1:])
    removed = 0
    for path in sorted(RESULTS.glob("*.csv")):
        if path.stem.startswith("baseline_") or path.stem in keep:
            continue
        path.unlink()
        removed += 1
        print("removed", path.name)
    if REPORTS.exists():
        shutil.rmtree(REPORTS)
        print("removed reports/")
    print("{} candidate file(s) removed; kept baselines + {}".format(removed, sorted(keep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
