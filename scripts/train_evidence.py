#!/usr/bin/env python3
"""Train the learned evidence model on the cached IDSSE matches and write it with its sha256.

    scripts/py scripts/train_evidence.py [--matches J03WPY ...] [--out PATH]

Recipe is fixed in possession/learned.py (seeded, single-threaded, noise seeds, stride). Runs the
leave-one-match-out fits only to choose the temperature, then fits on all matches. Default output:
possession.learned.default_model_path() under the work dir. Running it twice must print the same hash.
Prints aggregates only (shapes, timings, hash).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from possession import external, learned  # noqa: E402


def cached_matches() -> list[str]:
    """IDSSE ids with a prepared cache file (never triggers the slow XML parse)."""
    return [mid for mid in external.IDSSE_MATCHES if (external.cache_dir() / f"{mid}.npz").exists()]


def build_rows(ids: list[str]) -> dict:
    rows = {}
    for mid in ids:
        started = time.time()
        rows[mid] = learned.external_rows(external.load_external(mid))
        print("rows {}: {} x {} in {:.0f}s".format(mid, *rows[mid][0].shape, time.time() - started),
              flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    ids = args.matches or cached_matches()
    if len(ids) < 2:
        print("need at least 2 cached IDSSE matches, found {}".format(len(ids)))
        return 1
    started = time.time()
    final, _ = learned.fit_with_calibration(build_rows(ids))
    out = args.out or learned.default_model_path()
    learned.save_model(final, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print("matches={} temperature={} size_kb={:.0f} seconds={:.0f}".format(
        ",".join(ids), final.temperature, out.stat().st_size / 1024, time.time() - started))
    print("sha256 {}".format(digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
