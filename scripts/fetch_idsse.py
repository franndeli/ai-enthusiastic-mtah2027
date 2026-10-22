#!/usr/bin/env python3
"""Download the public IDSSE / Sportec Open DFL dataset (7 matches, CC-BY 4.0) into the work dir.

Public data (Bassek et al. 2025, Sci Data, doi:10.1038/s41597-025-04505-y), not match data: it
is only kept outside the repo because of its size. Files land in $MTAH_WORK_DIR/external/idsse/.
Prints names, sizes and SHA-256 only. Re-runnable: existing files with the right size are skipped.

    scripts/py scripts/fetch_idsse.py            # all seven matches, meta + tracking (+ event)
    scripts/py scripts/fetch_idsse.py --check    # verify what is there, download nothing
"""
import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# match_id -> figshare file ids, as hard-coded in kloppy 3.18 (kloppy._providers.sportec).
FILE_IDS = {
    "J03WPY": {"meta": 51643487, "event": 51643505, "tracking": 51643526},
    "J03WN1": {"meta": 51643472, "event": 51643496, "tracking": 51643517},
    "J03WMX": {"meta": 51643475, "event": 51643493, "tracking": 51643514},
    "J03WOH": {"meta": 51643478, "event": 51643499, "tracking": 51643520},
    "J03WQQ": {"meta": 51643484, "event": 51643508, "tracking": 51643529},
    "J03WOY": {"meta": 51643481, "event": 51643502, "tracking": 51643523},
    "J03WR9": {"meta": 51643490, "event": 51643511, "tracking": 51643532},
}
# The springernature.figshare.com mirror kloppy uses answers scripted requests with a 202 WAF
# challenge; the canonical ndownloader host (the API's own download_url) serves the file.
URL = "https://ndownloader.figshare.com/files/{}"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def idsse_dir() -> Path:
    work = os.environ.get("MTAH_WORK_DIR")
    if not work:
        sys.exit("MTAH_WORK_DIR is not set; run through scripts/py")
    path = Path(work) / "external" / "idsse"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(file_id: int, target: Path) -> str:
    request = urllib.request.Request(URL.format(file_id), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.status
        content_type = response.headers.get("Content-Type", "")
        if status != 200 or "html" in content_type:
            return "blocked (http {} {})".format(status, content_type)
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
    parser.add_argument("--kinds", nargs="+", default=["meta", "tracking", "event"])
    args = parser.parse_args()

    folder = idsse_dir()
    failures = 0
    for match_id, ids in FILE_IDS.items():
        for kind in args.kinds:
            target = folder / "{}_{}.xml".format(match_id, kind)
            if target.exists() and target.stat().st_size > 10_000:
                print("  present  {:<22} {:>12,d} B  sha256 {}".format(
                    target.name, target.stat().st_size, sha256(target)[:16]))
                continue
            if args.check:
                print("  MISSING  {}".format(target.name))
                failures += 1
                continue
            try:
                status = fetch(ids[kind], target)
            except Exception as error:  # network / WAF; never prints content
                status = "error {}".format(type(error).__name__)
            if status != "ok":
                failures += 1
                target.unlink(missing_ok=True)
                print("  FAILED   {:<22} {}".format(target.name, status))
            else:
                print("  fetched  {:<22} {:>12,d} B  sha256 {}".format(
                    target.name, target.stat().st_size, sha256(target)[:16]))
    print("\n{} file(s) missing or failed".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
