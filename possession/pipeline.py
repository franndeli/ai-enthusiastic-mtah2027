"""Regenerate the submission from the match data: `python run.py --out evaluation/results/<name>.csv`.

Reads the tactical ATD (and the Smart Tagging kick-offs) from MTAH_DATA_DIR, applies the logistic
coefficients in config/possession_v6.json (default; fitted on public data about other games only:
Metrica's sample games and the Sportec Open DFL matches), decodes H/A/D with switching costs, adds
PathCRF's ball-free team evidence inside the alive stretches and decodes again (v6; needs
requirements-v6.txt), writes `frame,label` (X outside the graded halves), runs the official
validator on that exact file and prints aggregates only. `--model config/possession_v5.json` (no
PathCRF, numpy and pandas only), `possession_v4.json` or `possession_v3.json` gives the earlier
versions. Deterministic: no randomness, no training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys

import numpy as np

from . import H, A, D, X, LABELS, __version__
from . import features as F
from . import smart_tagging
from .decode import segment
from .model import Model
from .reference import OFFICIAL, baseline_labels
from .tracking import read_epts

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "config", "possession_v6.json")
PRIVATE_ROOT = os.path.join(os.path.expanduser("~"), ".mtah2027")
V6_PACKAGES = ("torch", "torch_geometric", "scipy", "tqdm", "matplotlib")   # requirements-v6.txt


def missing_packages(names) -> list:
    """The names that are not installed, checked without importing them."""
    return [name for name in names if importlib.util.find_spec(name) is None]


def game_constants(game: str):
    sys.path.insert(0, os.path.abspath(OFFICIAL))
    import games  # the organisers' games.py
    g = games.GAMES[game]
    blocks = [(first, last + 1) for _, first, last in games.half_video_blocks(game)]
    return games, g["n_video_frames"], g["fps"], tuple(g["pitch"]), blocks


def write_csv(labels: np.ndarray, path: str) -> None:
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    letters = np.array(LABELS)[labels]
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (i, c) for i, c in enumerate(letters))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(labels: np.ndarray, blocks, fps: float) -> dict:
    graded = np.zeros(len(labels), dtype=bool)
    for a, b in blocks:
        graded[a:b] = True
    lab = labels[graded]
    spells = []
    for a, b in blocks:
        seg = labels[a:b]
        change = np.flatnonzero(np.diff(seg) != 0) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        for i in range(len(bounds) - 1):
            if seg[bounds[i]] in (H, A):
                spells.append(bounds[i + 1] - bounds[i])
    spells = np.array(spells) / fps
    return {"graded_frames": int(graded.sum()),
            "share_H": float(np.mean(lab == H)), "share_A": float(np.mean(lab == A)),
            "share_D": float(np.mean(lab == D)), "share_X": float(np.mean(lab == X)),
            "possessions": int(len(spells)),
            "mean_possession_s": float(spells.mean()) if len(spells) else 0.0,
            "median_possession_s": float(np.median(spells)) if len(spells) else 0.0,
            "possessions_under_1s": int(np.sum(spells < 1.0))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="H/A/D possession labels for the tactical video")
    parser.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR",
                                                         os.path.join(PRIVATE_ROOT, "data")),
                        help="data bundle root containing tactical/ (default $MTAH_DATA_DIR, "
                             "else ~/.mtah2027/data)")
    parser.add_argument("--game", default="BAR-ATH")
    parser.add_argument("--out", required=True,
                        help="output CSV, e.g. evaluation/results/<name>.csv (gitignored)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--baseline", type=int, default=None,
                        help="write the official baseline with this window instead (pipeline check)")
    parser.add_argument("--no-tags", action="store_true", help="ignore the Smart Tagging file")
    args = parser.parse_args(argv)

    model = None if args.baseline is not None else Model.from_json(args.model)
    if model is not None and model.p.get("pathcrf"):
        absent = missing_packages(V6_PACKAGES)
        if absent:
            print("%s needs PathCRF's packages, not installed here: %s. Install them with\n"
                  "  pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu\n"
                  "  pip install -r requirements-v6.txt\n"
                  "or run v5 instead: --model config/possession_v5.json"
                  % (os.path.basename(args.model), ", ".join(absent)))
            return 2

    games, n_frames, fps, pitch, blocks = game_constants(args.game)
    paths = games.asset_paths(args.game, data_dir=args.data)
    missing = [k for k in ("atd_meta", "atd_raw") if not os.path.isfile(paths[k])]
    if missing:
        print("match data not found: %s. Set MTAH_DATA_DIR to the folder that contains tactical/ "
              "(see DATA.md for the expected file names)." % ", ".join(missing))
        return 2
    print("reading the tactical ATD ...", flush=True)
    trk = read_epts(paths["atd_meta"], paths["atd_raw"], pitch=pitch, n_frames=n_frames)
    trk.blocks = blocks
    seen = ~np.isnan(trk.ball[:, 0])
    graded = np.zeros(trk.n, dtype=bool)
    for a, b in blocks:
        graded[a:b] = True
    objects = (~np.isnan(trk.players[graded, :, 0])).sum(axis=1)
    # Sanity aggregates to compare with DATA.md: 199,319 records, 46 slots, ball seen on 59.3% of
    # graded frames, ~20.5 tracked objects per graded frame.
    print("  %d frames, %d slots (%d home / %d away), ball seen on %.1f%% of graded frames, "
          "%.1f tracked objects per graded frame"
          % (trk.n, len(trk.team), int(np.sum(trk.team == 0)), int(np.sum(trk.team == 1)),
             100.0 * seen[graded].mean(), objects.mean()), flush=True)

    config = {"game": args.game, "version": __version__}
    if args.baseline is not None:
        labels = baseline_labels(trk, args.baseline, x_outside=False)
        config["mode"] = "official baseline, window %d" % args.baseline
    else:
        cfg = model.p.get("decode", {"cost_dead": 40.0, "cost_team": 8.0})
        feats = F.compute(trk)
        dead, team = model.dead_logit(feats), model.team_logit(feats)
        labels = segment(dead, team, blocks, trk.n,
                         cost_dead=cfg["cost_dead"], cost_team=cfg["cost_team"], fps=fps)
        config.update({"mode": "model", "model": os.path.basename(args.model), "decode": cfg})
        pcrf = model.p.get("pathcrf")
        if pcrf:
            # v6: ball-free team evidence from PathCRF inside the stretches decoded as alive,
            # then the same decode again (possession/pathcrf.py).
            from . import pathcrf
            print("running PathCRF (CPU, a few minutes) ...", flush=True)
            logodds, info = pathcrf.team_evidence(trk, labels, pcrf)
            labels = segment(dead, team + float(pcrf["weight"]) * logodds, blocks, trk.n,
                             cost_dead=cfg["cost_dead"], cost_team=cfg["cost_team"], fps=fps)
            config["pathcrf"] = dict(pcrf, **info)
            print("  PathCRF covered %.1f%% of the alive frames in %.0f s (torch %s)"
                  % (100 * info["coverage_of_alive"], info["seconds"],
                     sys.modules["torch"].__version__), flush=True)
        tags_path = paths.get("smart_tagging")
        if not args.no_tags and tags_path and os.path.isfile(tags_path):
            # Before each half's tagged kick-off window the ball is dead (the 2nd half is graded
            # from 110,808 but kicks off later). The tag only bounds a window; frames inside it
            # keep the decoded labels.
            wins = smart_tagging.kickoff_windows(smart_tagging.load(tags_path, fps))
            for a, b in blocks:
                inside = [w for w in wins if a - 60 * fps <= w[1] and w[0] < b]
                if inside and inside[0][0] > a:
                    labels[a:inside[0][0]] = D
            config["kickoff_tags"] = len(wins)
    write_csv(labels, args.out)
    digest = sha256(args.out)
    summary = stats(labels, blocks, fps)
    print("wrote %s  sha256 %s" % (args.out, digest))
    print("  possessions %d, mean %.1f s, median %.1f s, D %.1f%% of graded frames"
          % (summary["possessions"], summary["mean_possession_s"], summary["median_possession_s"],
             100 * summary["share_D"]))

    validator = os.path.join(os.path.abspath(OFFICIAL), "evaluation", "validate_submission.py")
    valid = None
    if os.path.isfile(validator):
        proc = subprocess.run([sys.executable, validator, args.out], capture_output=True, text=True)
        print(proc.stdout.strip())
        valid = proc.returncode == 0
    manifest = {"output": os.path.basename(args.out), "sha256": digest, "valid": valid,
                "config": config, "stats": summary,
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": __import__("pandas").__version__}
    if "pathcrf" in config:
        # Versions that matter when another machine has to regenerate the same SHA-256.
        from importlib import metadata
        manifest.update({name: metadata.version(name) for name in ("torch", "torch_geometric",
                                                                  "scipy")})
    work = os.environ.get("MTAH_WORK_DIR", os.path.join(PRIVATE_ROOT, "work"))
    os.makedirs(work, exist_ok=True)
    manifest_path = os.path.join(work, os.path.splitext(os.path.basename(args.out))[0]
                                 + "_manifest.json")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=True)
    print("manifest (aggregates only): %s" % manifest_path)
    return 0 if valid in (None, True) else 1


if __name__ == "__main__":
    sys.exit(main())
