"""Blind head-to-head of team pipelines on public matches none of them was trained on.

Each SkillCorner open-data match (A-League, broadcast tracking: ~41 % of graded frames without
the ball, like the challenge's ATD) is laid out as a full challenge bundle (possession/bundle.py);
every pipeline in the spec runs on it with its own, unchanged command; the official scorer
(evaluation/possession_metrics.py) scores each CSV. Public data only: prints aggregates only.

usage: python scripts/rehearsal.py SPEC.json --skillcorner <opendata>/data/matches \
           --sc-tracking DIR --work DIR [--sc-cache DIR] [--matches ID ...] [--jobs 2]

SPEC.json is a list of pipelines; ``{data}`` (bundle folder), ``{out}`` (CSV to write),
``{outdir}``, ``{tmp}`` (fresh empty folder) and ``{python}`` are filled in:

    [{"name": "v5", "cwd": "/path/to/repo", "cmd": ["{python}", "run.py", "--data", "{data}",
      "--out", "{out}"]},
     {"name": "other", "cwd": "/path/to/other", "cmd": ["{python}", "run.py", "--out", "{out}"],
      "env": {"MTAH_DATA_DIR": "{data}", "MTAH_WORK_DIR": "{tmp}"}}]

``"result"`` overrides where the CSV lands (default ``{out}``). Per-match results are appended to
WORK/results.jsonl, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import bundle  # noqa: E402
from possession.skillcorner import load_match  # noqa: E402


def fill(value, ctx):
    if isinstance(value, str):
        return value.format(**ctx)
    if isinstance(value, list):
        return [fill(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: fill(v, ctx) for k, v in value.items()}
    return value


def run_one(spec: dict, data: str, work: str) -> tuple:
    """(csv path or None, seconds, error tail)."""
    outdir = os.path.join(work, "out", spec["name"])
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir)
    tmp = tempfile.mkdtemp(prefix="rehearsal_%s_" % spec["name"], dir=work)
    ctx = {"data": data, "outdir": outdir, "out": os.path.join(outdir, "pred.csv"), "tmp": tmp,
           "python": sys.executable}
    env = dict(os.environ)
    env.update(fill(spec.get("env", {}), ctx))
    started = time.time()
    with open(os.path.join(work, "log_%s.txt" % spec["name"]), "w") as log:
        proc = subprocess.run(fill(spec["cmd"], ctx), cwd=fill(spec.get("cwd", ROOT), ctx),
                              env=env, stdout=log, stderr=subprocess.PIPE, text=True)
    shutil.rmtree(tmp, ignore_errors=True)
    result = fill(spec.get("result", "{out}"), ctx)
    if proc.returncode != 0 or not os.path.isfile(result):
        return None, time.time() - started, (proc.stderr or "")[-600:]
    return result, time.time() - started, ""


def score(gt: np.ndarray, path: str) -> dict:
    table = pm.read_label_csv(path)
    problem = pm.validate_submission(table, len(gt))
    if problem:
        return {"error": problem}
    r = pm.evaluate(gt, pm.label_codes(table))
    cm = np.array(r["confusion"])[:pm.N_SCORED]
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "poss": r["n_pred_spells"],
            "gt_poss": r["n_gt_spells"], "alive_dead_f1": r["alive_dead_f1"],
            "team_acc": r["team_acc_both_alive"], "d_share": float(cm[:, pm.D].sum() / cm.sum()),
            "gt_d_share": float(cm[pm.D].sum() / cm.sum())}


def summary(rows: list, names: list) -> None:
    done = [r for r in rows if all("spellF1" in r["scores"].get(n, {}) for n in names)]
    if not done:
        return
    print("\n%d matches scored by every pipeline (official spellF1, macroF1)" % len(done))
    print("  %-14s %8s %8s %10s %9s %9s %12s" % ("pipeline", "spellF1", "macroF1", "poss/gt",
                                                 "alive/dead", "team acc", "D share/gt"))
    gt_poss = np.mean([r["scores"][names[0]]["gt_poss"] for r in done])
    gt_d = np.mean([r["scores"][names[0]].get("gt_d_share", np.nan) for r in done])

    def mean(n, key):
        return np.mean([r["scores"][n].get(key, np.nan) for r in done])
    for n in names:
        print("  %-14s %8.4f %8.4f %5.0f/%-4.0f %9.3f %9.3f %6.1f/%.1f %%"
              % (n, mean(n, "spellF1"), mean(n, "macroF1"), mean(n, "poss"), gt_poss,
                 mean(n, "alive_dead_f1"), mean(n, "team_acc"), 100 * mean(n, "d_share"),
                 100 * gt_d))
    rng = np.random.RandomState(0)
    draws = rng.randint(0, len(done), size=(2000, len(done)))
    ref = names[0]
    for n in names[1:]:
        d = np.array([r["scores"][ref]["spellF1"] - r["scores"][n]["spellF1"] for r in done])
        lo, hi = np.percentile(d[draws].mean(axis=1), [2.5, 97.5])
        print("  %s - %s: %+.4f [%+.4f, %+.4f], %s better on %d of %d matches"
              % (ref, n, d.mean(), lo, hi, ref, int((d > 0).sum()), len(d)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec")
    ap.add_argument("--skillcorner", required=True, help="<opendata>/data/matches")
    ap.add_argument("--sc-tracking", required=True, help="folder with <id>_tracking.jsonl")
    ap.add_argument("--sc-cache", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--matches", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--keep", action="store_true", help="keep the last bundle on disk")
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as handle:
        specs = json.load(handle)
    names = [s["name"] for s in specs]
    os.makedirs(args.work, exist_ok=True)
    res_path = os.path.join(args.work, "results.jsonl")
    rows = []
    if os.path.isfile(res_path):
        with open(res_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    done = {r["match"] for r in rows if all(n in r["scores"] for n in names)}
    rows = [r for r in rows if r["match"] in done]
    matches = args.matches or sorted(os.listdir(args.skillcorner))
    lay = bundle.layout()
    data = os.path.join(args.work, "bundle")
    print("%-9s %7s %6s  " % ("match", "ball", "gt") + "  ".join("%18s" % n for n in names))
    for mid in matches:
        if mid in done:
            continue
        trk, lab = load_match(mid, os.path.join(args.skillcorner, mid),
                              os.path.join(args.sc_tracking, "%s_tracking.jsonl" % mid),
                              args.sc_cache)
        arrays = bundle.place(trk, lab, lay, seed=int(mid) % 10000)
        shutil.rmtree(data, ignore_errors=True)
        bundle.write(data, arrays, lay, "SC-%s" % mid)
        gt = arrays["gt"]
        graded = gt != pm.X
        missing = float(np.isnan(arrays["ball"][graded, 0]).mean())
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            outs = list(pool.map(lambda s: run_one(s, data, args.work), specs))
        scores, cells = {}, []
        for spec, (path, secs, err) in zip(specs, outs):
            if path is None:
                scores[spec["name"]] = {"error": err.strip().splitlines()[-1] if err.strip() else
                                        "failed"}
                cells.append("%18s" % "FAILED")
                print("  %s failed on %s:\n%s" % (spec["name"], mid, err), file=sys.stderr)
                continue
            s = score(gt, path)
            s["seconds"] = round(secs, 1)
            scores[spec["name"]] = s
            cells.append("%18s" % ("%.4f (%d) %4.0fs" % (s["spellF1"], s["poss"], secs)
                                   if "spellF1" in s else s["error"][:18]))
        row = {"match": mid, "ball_missing": missing,
               "gt_poss": int(pm.possession_stats(gt, pm.valid_runs(gt))["n_spells"]),
               "scores": scores}
        rows.append(row)
        with open(res_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print("%-9s %6.1f%% %6d  " % (mid, 100 * missing, row["gt_poss"]) + "  ".join(cells),
              flush=True)
    if not args.keep:
        shutil.rmtree(data, ignore_errors=True)
    summary(rows, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
