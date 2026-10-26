#!/usr/bin/env python3
"""Download the public Metrica Sports sample data (3 anonymised matches) into the work dir.

Public data (github.com/metrica-sports/sample-data, 25 fps tracking normalised 0-1 on 105x68 m
plus synchronised event data), not match data: it is kept outside the repo only because of its
size. Files land in $MTAH_WORK_DIR/external/metrica/. Prints names, sizes and SHA-256 only.

    scripts/py scripts/fetch_metrica.py            # all three games
    scripts/py scripts/fetch_metrica.py --check    # verify what is there, download nothing
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
FILES = {
    "Sample_Game_1": ["Sample_Game_1_RawEventsData.csv",
                      "Sample_Game_1_RawTrackingData_Home_Team.csv",
                      "Sample_Game_1_RawTrackingData_Away_Team.csv"],
    "Sample_Game_2": ["Sample_Game_2_RawEventsData.csv",
                      "Sample_Game_2_RawTrackingData_Home_Team.csv",
                      "Sample_Game_2_RawTrackingData_Away_Team.csv"],
    "Sample_Game_3": ["Sample_Game_3_events.json", "Sample_Game_3_metadata.xml",
                      "Sample_Game_3_tracking.txt"],
}
USER_AGENT = "mtah2027-fetch-metrica/1.0 (python-urllib)"


def metrica_dir() -> Path:
    work = os.environ.get("MTAH_WORK_DIR")
    if not work:
        sys.exit("MTAH_WORK_DIR is not set; run through scripts/py")
    path = Path(work) / "external" / "metrica"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.status != 200:
            return "failed (http {})".format(response.status)
        with open(target, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is present")
    args = parser.parse_args()
    folder = metrica_dir()
    failures = 0
    for game, names in FILES.items():
        for name in names:
            target = folder / name
            if target.exists() and target.stat().st_size > 10_000:
                status = "present"
            elif args.check:
                status, failures = "missing", failures + 1
            else:
                try:
                    status = fetch(BASE + game + "/" + name, target)
                except Exception as error:  # network errors carry no data
                    status = "failed ({})".format(type(error).__name__)
                if status != "ok":
                    failures += 1
            size = target.stat().st_size if target.exists() else 0
            digest = sha256(target)[:16] if size else "-"
            print("{:50s} {:>12,d} bytes  {}  {}".format(name, size, digest, status))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
