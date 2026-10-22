#!/usr/bin/env python3
"""Byte-compare two generated submission files and report sizes and SHA-256 only (never content).

    scripts/py scripts/check_repro.py evaluation/results/a.csv evaluation/results/b.csv
"""
import hashlib
import sys
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    da, db = digest(a), digest(b)
    print("{}  {:>10,d} B  sha256 {}".format(a.name, a.stat().st_size, da[:16]))
    print("{}  {:>10,d} B  sha256 {}".format(b.name, b.stat().st_size, db[:16]))
    print("IDENTICAL" if da == db else "DIFFERENT")
    return 0 if da == db else 1


if __name__ == "__main__":
    sys.exit(main())
