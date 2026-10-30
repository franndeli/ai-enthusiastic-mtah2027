"""See what v6 does without the challenge data, in the exact submission path. Aggregates only.

A public Sportec Open DFL match (default J03WR9, which PathCRF never trained on) is laid out like
the challenge's files (possession/bundle.py: ATD text format, the challenge's 199,319-frame
layout, tracking degraded like the ATD: ball missing on ~41 % of frames, dropouts, identity swaps,
no position jitter), with its labels as the ground truth. Then `run.py` runs on it with v5 and
with v6 (twice) and the official scorer compares each output with the labels.

Checks: every output valid; v6 gives the same SHA-256 twice; dead ball identical in v5 and v6; v6
scores above v5. The SHA-256 values can be compared between machines (PIPELINE.md lists the
expected ones); the input file's SHA-256 is printed so that a different input can be told apart
from a different result. The shipped v5 was trained on all seven DFL matches, J03WR9 included,
which favours v5 here. About 10 minutes on two CPU cores; needs requirements-v6.txt.

usage: python scripts/e2e_public.py <dfl dir> [--match J03WR9] [--seed 10] [--work DIR]
       <dfl dir>: dfl_<match>.npz from scripts/fetch_dfl.py
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import D, bundle  # noqa: E402
from possession.dfl import controlled, load as load_dfl  # noqa: E402
from possession.external import degrade  # noqa: E402
from possession.reference import metrics  # noqa: E402

RUNS = (("v5", ["--model", os.path.join("config", "possession_v5.json")]),
        ("v6", []), ("v6 again", []))


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dfl", help="folder with dfl_<match>.npz (scripts/fetch_dfl.py)")
    parser.add_argument("--match", default="J03WR9")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--work", default=os.path.join(os.path.expanduser("~"), ".mtah2027", "work",
                                                       "e2e_public"))
    args = parser.parse_args()
    source = os.path.join(args.dfl, "dfl_%s.npz" % args.match)
    data = os.path.join(args.work, "bundle_%s_seed%d" % (args.match, args.seed))
    print("input  %s  sha256 %s" % (os.path.basename(source), sha256(source)[:16]))

    trk, raw = load_dfl(source)
    labels = controlled(raw, trk.blocks, trk.fps)
    degraded = degrade(trk, labels, noise={"player_noise_m": 0.0}, seed=args.seed)
    lay = bundle.layout()
    arrays = bundle.place(degraded, labels, lay, seed=args.seed)
    paths = bundle.write(data, arrays, lay, "DFL-%s" % args.match)
    gt = arrays["gt"]
    print("bundle %s  (ATD sha256 %s)" % (data, sha256(paths["atd_raw"])[:16]), flush=True)

    pm = metrics()
    results = []
    for name, extra in RUNS:
        out = os.path.join(args.work, "%s_%s.csv" % (args.match, name.replace(" ", "_")))
        print("run.py %s ..." % name, flush=True)
        proc = subprocess.run([sys.executable, os.path.join(REPO, "run.py"), "--data", data,
                               "--out", out] + extra, cwd=REPO, capture_output=True, text=True,
                              env=dict(os.environ, MTAH_WORK_DIR=args.work))
        if proc.returncode != 0:
            print(proc.stdout[-1500:] + proc.stderr[-1500:])
            print("FAIL: run.py %s exited with %d" % (name, proc.returncode))
            return 1
        pred = pm.label_codes(pm.read_label_csv(out))
        r = pm.evaluate(gt, pred)
        stem = os.path.splitext(os.path.basename(out))[0]
        with open(os.path.join(args.work, stem + "_manifest.json"), encoding="utf-8") as handle:
            share_d = json.load(handle)["stats"]["share_D"]      # as run.py prints it
        results.append({"name": name, "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                        "poss": r["n_pred_spells"], "truth": r["n_gt_spells"], "D": share_d,
                        "dead": pred == D, "valid": "Valid" in proc.stdout, "sha": sha256(out)})

    print("\n| run | spellF1 | macroF1 | possessions (truth %d) | D share | sha256 |" % results[0]["truth"])
    print("|---|---|---|---|---|---|")
    for r in results:
        print("| %s | %.4f | %.4f | %d | %.1f %% | %s |" % (r["name"], r["spellF1"], r["macroF1"],
                                                         r["poss"], 100 * r["D"], r["sha"][:16]))
    v5, v6, again = results
    checks = [("every output valid", all(r["valid"] for r in results)),
              ("v6 gives the same SHA-256 twice", v6["sha"] == again["sha"]),
              ("dead ball identical in v5 and v6", bool(np.array_equal(v5["dead"], v6["dead"]))),
              ("v6 above v5 (spellF1)", v6["spellF1"] > v5["spellF1"])]
    print()
    for text, ok in checks:
        print("  %s  %s" % ("ok  " if ok else "FAIL", text))
    summary = {r["name"]: {k: r[k] for k in ("spellF1", "macroF1", "poss", "D", "sha")}
               for r in results}
    with open(os.path.join(args.work, "e2e_%s_seed%d.json" % (args.match, args.seed)), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
