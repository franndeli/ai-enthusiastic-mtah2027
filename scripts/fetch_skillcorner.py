#!/usr/bin/env python3
"""Download the SkillCorner open data (20 A-League 2024/25 matches, MIT licence) into the work dir.

Public data (github.com/SkillCorner/opendata at a pinned commit): broadcast tracking at 10 fps
with the provider's per-frame possession, dynamic events and phases of play. Not match data; it
is kept outside the repo only because of its size (1.9 GB). Every file is checked against the
size and hash recorded in config/skillcorner_manifest.json (git blob SHA-1, or the Git LFS
SHA-256 for the tracking). Prints names, sizes and verification status only.

    scripts/py scripts/fetch_skillcorner.py                 # all 20 matches
    scripts/py scripts/fetch_skillcorner.py --check         # verify what is there, download nothing
    scripts/py scripts/fetch_skillcorner.py --make-manifest # rebuild the manifest from GitHub
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "config" / "skillcorner_manifest.json"
REPO = "SkillCorner/opendata"
COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
KINDS = {"match.json": "match.json", "dynamic_events.csv": "dynamic_events.csv",
         "phases_of_play.csv": "phases.csv", "tracking_extrapolated.jsonl": "tracking.jsonl"}
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/{COMMIT}/"
USER_AGENT = "mtah2027-fetch-skillcorner/1.0 (python-urllib)"


def skillcorner_dir() -> Path:
    work = os.environ.get("MTAH_WORK_DIR")
    if not work:
        sys.exit("MTAH_WORK_DIR is not set; run through scripts/py")
    return Path(work) / "external" / "skillcorner" / COMMIT


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def make_manifest() -> dict:
    tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
    if tree.get("truncated"):
        raise SystemExit("GitHub tree listing truncated")
    files = []
    for item in tree["tree"]:
        parts = item["path"].split("/")
        if item["type"] != "blob" or len(parts) != 4 or parts[:2] != ["data", "matches"]:
            continue
        mid, name = parts[2], parts[3]
        kind = name[len(mid) + 1:] if name.startswith(mid + "_") else None
        if kind not in KINDS:
            continue
        entry = {"match": mid, "kind": kind, "path": item["path"]}
        if item["size"] < 400:      # a Git LFS pointer: the real size and hash are inside it
            pointer = dict(line.split(" ", 1) for line in _get(RAW + item["path"]).decode().splitlines() if line)
            entry.update(lfs=True, size=int(pointer["size"]), algorithm="sha256",
                         digest=pointer["oid"].split(":", 1)[1])
        else:
            entry.update(lfs=False, size=int(item["size"]), algorithm="sha1", digest=item["sha"])
        files.append(entry)
    files.sort(key=lambda e: (e["match"], e["kind"]))
    return {"repository": f"https://github.com/{REPO}", "commit": COMMIT, "licence": "MIT",
            "files": files}


def digest_of(path: Path, algorithm: str, size: int) -> str:
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":         # git blob hash
        digest.update(b"blob " + str(size).encode("ascii") + b"\0")
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path: Path, entry: dict) -> bool:
    return (path.is_file() and path.stat().st_size == entry["size"]
            and digest_of(path, entry["algorithm"], entry["size"]) == entry["digest"])


def fetch(entry: dict, target: Path) -> str:
    url = (MEDIA if entry["lfs"] else RAW) + entry["path"]
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
        temporary = Path(out.name)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    out.write(chunk)
            out.flush()
            if not verified(temporary, entry):
                return "failed (size or hash mismatch)"
            temporary.replace(target)
            return "ok"
        finally:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report what is present")
    parser.add_argument("--make-manifest", action="store_true")
    args = parser.parse_args()
    if args.make_manifest:
        manifest = make_manifest()
        MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
        matches = sorted({e["match"] for e in manifest["files"]})
        print(f"manifest: {len(manifest['files'])} files, {len(matches)} matches, "
              f"{sum(e['size'] for e in manifest['files']):,d} bytes")
        return 0
    manifest = json.loads(MANIFEST.read_text())
    if manifest["commit"] != COMMIT:
        raise SystemExit("manifest commit differs from the pinned commit")
    folder = skillcorner_dir()
    failures = 0
    for entry in manifest["files"]:
        target = folder / entry["match"] / KINDS[entry["kind"]]
        if verified(target, entry):
            status = "present"
        elif args.check:
            status, failures = "missing", failures + 1
        else:
            try:
                status = fetch(entry, target)
            except Exception as error:  # network errors carry no data
                status = "failed ({})".format(type(error).__name__)
            if status != "ok":
                failures += 1
        print("{} {:28s} {:>12,d} bytes  {}".format(entry["match"], entry["kind"], entry["size"], status),
              flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
