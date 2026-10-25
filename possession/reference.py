"""The official baseline algorithm (starter/baseline.py) on in-memory tracking, plus the official
metric, both loaded from the organisers' repository so nothing is re-implemented by hand."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from . import H, A, D, X
from .tracking import HOME, AWAY, Tracking

def _find_official() -> str:
    """The organisers' repository: $MTAH_OFFICIAL_REPO, a third_party/mtah27 submodule, or the
    enclosing directory when this package lives inside a fork of the official repository."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("MTAH_OFFICIAL_REPO"),
                  os.path.join(here, "..", "third_party", "mtah27"),
                  os.path.join(here, ".."), os.path.join(here, "..", "..")]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "evaluation", "possession_metrics.py")):
            return os.path.abspath(path)
    return os.path.abspath(candidates[1])


OFFICIAL = _find_official()


def _load(name: str, relpath: str):
    path = os.path.abspath(os.path.join(OFFICIAL, relpath))
    if name in sys.modules:
        return sys.modules[name]
    root = os.path.abspath(OFFICIAL)
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metrics():
    return _load("possession_metrics", "evaluation/possession_metrics.py")


def official_baseline():
    return _load("mtah_official_baseline", "starter/baseline.py")


def nearest_team_codes(trk: Tracking) -> np.ndarray:
    """Step 1 of the official baseline: team of the nearest tracked player, -1 without ball."""
    bl = official_baseline()
    out = np.full(trk.n, bl.NOBODY, dtype=np.int8)
    seen = ~np.isnan(trk.ball[:, 0])
    d = np.linalg.norm(trk.players.astype(np.float64) - trk.ball.astype(np.float64)[:, None, :],
                       axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    best = np.argmin(d, axis=1)
    has = seen & np.isfinite(d[np.arange(trk.n), best])
    out[has] = np.where(trk.team[best[has]] == HOME, bl.HOME, bl.AWAY)
    return out


def baseline_labels(trk: Tracking, window: int, x_outside: bool = True) -> np.ndarray:
    """Official baseline labels (codes). With x_outside=False the output is exactly the official
    file, which also writes D on ungraded frames."""
    bl = official_baseline()
    pm = metrics()
    letters = bl.smooth(nearest_team_codes(trk), window)
    codes = pm.label_codes(letters).astype(np.int8)
    if not x_outside:
        return codes
    inside = np.zeros(trk.n, dtype=bool)
    for a, b in trk.blocks:
        inside[a:b] = True
    codes[~inside] = X
    return codes


def score(gt: np.ndarray, pred: np.ndarray) -> dict:
    pm = metrics()
    r = pm.evaluate(gt.astype(np.int8), pred.astype(np.int8))
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "poss": r["n_pred_spells"], "gt_poss": r["n_gt_spells"],
            "alive_dead_f1": r["alive_dead_f1"], "team_acc": r["team_acc_both_alive"],
            "lat25": r["transitions"][25]["mean_latency"]}


def spell_flags(gt: np.ndarray, pred: np.ndarray):
    """(flags (n_thresholds, n_gt), n_pred) -- matched GT possessions per IoU threshold."""
    pm = metrics()
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred.astype(np.int8), runs)
    gts = pm.possession_spells(gt, runs)
    prs = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gts, prs)
    flags = np.vstack([pm.matched_gt_flags(ov, len(gts), t) for t in pm.SPELL_IOUS])
    return flags, len(prs)


def paired_bootstrap(cases, n_draws: int = 1000, seed: int = 0) -> dict:
    """cases: list of (gt, pred_a, pred_b). Resamples GT possessions within each case, pools
    TP and counts across cases, returns the mean and 5-95% interval of spellF1(A) - spellF1(B)."""
    rng = np.random.RandomState(seed)
    prepared = []
    for gt, pa, pb in cases:
        fa, na = spell_flags(gt, pa)
        fb, nb = spell_flags(gt, pb)
        prepared.append((fa, na, fb, nb))

    def f1(tp, n_pred, n_gt):
        fp = np.maximum(0, n_pred - tp)          # same clamp as possession_metrics' bootstrap
        denom = 2 * tp + fp + (n_gt - tp)
        return np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))

    def pooled(draw):
        tpa = np.zeros(5)
        tpb = np.zeros(5)
        ngt = npa = npb = 0
        for (fa, na, fb, nb), picks in zip(prepared, draw):
            tpa += fa[:, picks].sum(axis=1)
            tpb += fb[:, picks].sum(axis=1)
            ngt += len(picks)
            npa += na
            npb += nb
        return f1(tpa, npa, ngt) - f1(tpb, npb, ngt)

    point = pooled([np.arange(p[0].shape[1]) for p in prepared])
    diffs = np.empty(n_draws)
    for k in range(n_draws):
        diffs[k] = pooled([rng.randint(0, p[0].shape[1], p[0].shape[1]) for p in prepared])
    lo, hi = np.percentile(diffs, [5, 95])
    return {"delta": float(point), "low": float(lo), "high": float(hi)}
