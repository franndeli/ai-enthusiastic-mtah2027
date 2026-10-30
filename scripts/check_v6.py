"""Check that v6 runs on this machine and reproduces the team's reference output on the match.

1. Python 3.11 and the package versions pinned in requirements-v6.txt.
2. Runs `python run.py --out evaluation/results/v6.csv` (the submission path; it prints aggregates
   only), unless --no-run, which checks an existing file instead.
3. Compares the file's SHA-256 with config/reference_outputs.json (Ángel's machine, 30 Oct).

Ends with PASS or FAIL and a short report to paste in the team chat (versions, SHA-256,
possessions, dead-ball share). Nothing leaves the machine.

usage: python scripts/check_v6.py [--data DIR] [--out evaluation/results/v6.csv] [--no-run]
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = os.path.join(REPO, "requirements-v6.txt")
REFERENCE = os.path.join(REPO, "config", "reference_outputs.json")
MODEL = "possession_v6.json"
INSTALL = ("pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu\n"
           "  pip install -r requirements-v6.txt")


def read_pins(path: str = PINS) -> dict:
    """{package: version} from the `name==version` lines of a requirements file."""
    pins = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line.split("#", 1)[0].strip())
            if match:
                pins[match.group(1)] = match.group(2)
    return pins


def installed(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def compare(pins: dict, found: dict) -> dict:
    """{package: 'ok' | 'missing' | 'different'}; build labels such as +cpu are ignored."""
    status = {}
    for name, want in pins.items():
        have = found.get(name)
        status[name] = ("missing" if have is None else
                        "ok" if have.split("+")[0] == want else "different")
    return status


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_stats(out: str) -> dict:
    work = os.environ.get("MTAH_WORK_DIR", os.path.join(os.path.expanduser("~"), ".mtah2027",
                                                        "work"))
    path = os.path.join(work, os.path.splitext(os.path.basename(out))[0] + "_manifest.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("stats", {})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None,
                        help="folder that contains tactical/ (default: run.py's, i.e. "
                             "$MTAH_DATA_DIR or ~/.mtah2027/data)")
    parser.add_argument("--out", default=os.path.join("evaluation", "results", "v6.csv"))
    parser.add_argument("--no-run", action="store_true", help="only check an existing --out file")
    parser.add_argument("--game", default="BAR-ATH")
    parser.add_argument("--expect", default=None,
                        help="compare with this SHA-256 instead of the reference (tests)")
    args = parser.parse_args(argv)
    out = args.out if os.path.isabs(args.out) else os.path.join(REPO, args.out)

    print("1. environment (pinned in requirements-v6.txt)")
    py_ok = sys.version_info[:2] == (3, 11)
    print("   %-16s %-14s %s" % ("python", platform.python_version(),
                                 "ok" if py_ok else "expected 3.11"))
    pins = read_pins()
    found = {name: installed(name) for name in pins}
    status = compare(pins, found)
    for name, want in pins.items():
        note = "ok" if status[name] == "ok" else "%s (pinned %s)" % (status[name], want)
        print("   %-16s %-14s %s" % (name, found[name] or "-", note))
    if "missing" in status.values():
        print("\nFAIL: packages missing. In a Python 3.11 venv, from the repository root:\n  "
              + INSTALL)
        return 2

    if args.expect:
        expected, source = args.expect, "--expect"
        ref = {}
    else:
        with open(REFERENCE, encoding="utf-8") as handle:
            ref = json.load(handle)[args.game][MODEL]
        expected, source = ref["sha256"], ref["reference_machine"]

    if not args.no_run:
        print("\n2. run.py (aggregates only; a few minutes on CPU)", flush=True)
        cmd = [sys.executable, os.path.join(REPO, "run.py"), "--out", out, "--game", args.game]
        if args.data:
            cmd += ["--data", args.data]
        code = subprocess.call(cmd, cwd=REPO)
        if code != 0:
            print("\nFAIL: run.py stopped with exit code %d (its message is above)." % code)
            return 2 if code == 2 else 1
    if not os.path.isfile(out):
        print("\nFAIL: %s does not exist; run without --no-run." % out)
        return 2

    digest = sha256(out)
    stats = manifest_stats(out)
    passed = digest == expected
    print("\n3. result\n")
    print("check_v6 %s" % ("PASS" if passed else "FAIL"))
    print("  sha256       %s" % digest)
    print("  reference    %s (%s)" % (expected, source))
    if stats:
        line = "  output       %d possessions, D %.1f %%" % (stats["possessions"],
                                                             100 * stats["share_D"])
        if ref:
            line += " (reference %d, %.1f %%)" % (ref["possessions"], 100 * ref["share_D"])
        print(line)
    print("  machine      %s, Python %s" % (platform.platform(), platform.python_version()))
    print("  packages     %s" % ", ".join("%s %s" % (n, found[n]) for n in pins))
    if passed:
        print("\nSame file as the reference: this machine regenerates the submission exactly.")
        return 0
    print("\nDifferent file. Paste the report above in the team chat; see PIPELINE.md, "
          "'If the check fails'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
