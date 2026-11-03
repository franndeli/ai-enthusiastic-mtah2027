#!/usr/bin/env python3
"""BR2 external-only, whole-match-excluded evaluation; aggregate JSON logs only.
Real-match runs require RunPod. Protocol: docs/research/tracker_robust_dead.md.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from evaluation import possession_metrics as pm
from possession import D, H, A, evidence, external, features, metrica, robust_dead as rd
from possession.io import work_dir
from run import RULES_CONFIG as DEFAULT_CONFIG, decode_blocks, load_config

DEV = tuple("idsse:" + m for m in ("J03WMX", "J03WN1", "J03WOH", "J03WOY")) + (
    "metrica:Sample_Game_1", "metrica:Sample_Game_2")
CONFIRM = tuple("idsse:" + m for m in ("J03WPY", "J03WQQ", "J03WR9")) + ("metrica:Sample_Game_3",)
SCENARIOS = ("standard", "smooth", "jitter", "drift")
SHAPE_KEYS = ("len", "width", "speed", "vel")
TRAIN_SEEDS, DEV_SEEDS, CONFIRM_SEEDS = (0, 1), (11, 29), (101, 307, 911)


def provider(mid: str) -> str:
    return mid.split(":", 1)[0]


def load_match(mid: str):
    if mid not in DEV + CONFIRM:
        raise ValueError("match outside fixed public-external allowlist")
    source, name = mid.split(":", 1)
    ext = external.load_external(name) if source == "idsse" else metrica.load_game(name)
    if ext.fps != 25:
        raise ValueError("unexpected external frame rate")
    rd.blocks(ext.match)
    return ext


def perturb(match, source: str, scenario: str, seed: int):
    """Label-free paired noise; existing ball gaps retained, no invented dead trajectories."""
    if source not in ("idsse", "metrica") or scenario not in SCENARIOS:
        raise ValueError("unknown noise protocol")
    params = external.NoiseParams(ball_missing_share=.41 if source == "idsse" else .15)
    noisy = external.inject_noise(match, seed, params)
    if scenario == "standard":
        return noisy
    rng = np.random.RandomState(seed + 2317)
    xy = noisy.players_xy.copy()
    scale = np.asarray(noisy.pitch_m)
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    sigma = .03 if scenario == "smooth" else .30
    for a, b in rd.blocks(noisy):
        noise = rng.normal(0., sigma, xy[a:b].shape)
        if scenario == "drift":
            noise = uniform_filter1d(noise, size=25, axis=0, mode="reflect") * np.sqrt(25.)
        xy[a:b] += (noise / scale).astype(np.float32)
        median = 75. / np.exp(.5 * (np.log(2.) / external.Z_P90) ** 2)
        for slot in range(xy.shape[1]):
            dropped = external.dropout_mask(b - a, .07, median, 2 * median, rng)
            xy[a:b, slot][dropped] = np.nan
        # Match a short-span provider speed channel, independently inside each half.
        local = xy[a:b].astype(np.float64) * scale
        if b - a > 2:
            v = np.linalg.norm(local[2:] - local[:-2], axis=2) * 12.5
            valid = np.isfinite(local[1:-1]).all(axis=2)
            speed[a + 1:b - 1] = np.where(valid, v, np.nan)
    return replace(noisy, players_xy=xy, players_speed=speed)


def conditions(seeds):
    return [("clean", None)] + [(s, seed) for s in SCENARIOS for seed in seeds]


def fingerprint() -> str:
    h = hashlib.sha256()
    for path in (ROOT / "possession/robust_dead.py", Path(__file__), DEFAULT_CONFIG,
                 ROOT / "possession/features.py", ROOT / "possession/evidence.py",
                 ROOT / "possession/external.py", ROOT / "possession/metrica.py",
                 ROOT / "run.py", ROOT / "evaluation/possession_metrics.py"):
        h.update(path.read_bytes())
    return h.hexdigest()


def training_rows(mid: str, cfg: dict, cache: Path) -> dict:
    """Cache sampled training observations only; caller supplies explicit training IDs."""
    path = cache / (mid.replace(":", "_") + ".npz")
    if path.exists():
        with np.load(path) as blob:
            return {k: blob[k] for k in blob.files}
    ext = load_match(mid)
    rows = np.flatnonzero(ext.match.graded_mask() & np.isin(ext.codes, (H, A, D)))[::5]
    if not len(rows):
        raise ValueError("empty training match")
    xs, ys, offsets, weights = [], [], [], []
    for scenario, seed in conditions(TRAIN_SEEDS):
        match = ext.match if seed is None else perturb(ext.match, provider(mid), scenario, seed)
        feats = features.compute(match)
        xs.append(rd.player_design(match)[rows])
        ys.append((ext.codes[rows] == D).astype(np.int8))
        offsets.append(rd.rule_offset(feats, cfg["evidence"])[rows])
        weights.append(np.full(len(rows), 1. / (len(rows) * len(conditions(TRAIN_SEEDS)))))
    shape = features.ball_free_shape_features(ext.match)
    shape_x = np.column_stack([shape["shape_" + k + "_z"][rows] for k in SHAPE_KEYS])
    shape_y = np.where(ext.codes[rows] == H, 1, np.where(ext.codes[rows] == A, 0, -1))
    shape_y[~np.isfinite(shape_x).all(axis=1)] = -1
    result = dict(x=np.concatenate(xs), y=np.concatenate(ys), offset=np.concatenate(offsets),
                  weight=np.concatenate(weights), shape_x=shape_x, shape_y=shape_y)
    np.savez_compressed(path, **result)
    return result


def fit_fold(train_ids: tuple[str, ...], evaluation_ids: tuple[str, ...], cfg: dict, cache: Path):
    """All fitting and standardisation use only these explicit training matches."""
    if not train_ids or len(set(train_ids)) != len(train_ids) or set(train_ids) & set(evaluation_ids):
        raise ValueError("invalid fit/evaluation exclusion boundary")
    if not set(train_ids + evaluation_ids) <= set(DEV + CONFIRM):
        raise ValueError("unapproved fit/evaluation match")
    rows = [(mid, training_rows(mid, cfg, cache)) for mid in train_ids]
    sources = {provider(mid) for mid in train_ids}
    count = {s: sum(provider(mid) == s for mid in train_ids) for s in sources}
    weights = [r["weight"] / (len(sources) * count[provider(mid)]) for mid, r in rows]
    model = rd.fit_activity(np.concatenate([r["x"] for _, r in rows]),
                            np.concatenate([r["y"] for _, r in rows]),
                            np.concatenate([r["offset"] for _, r in rows]), np.concatenate(weights))
    # Refit the baseline's shape learner too. No pooled historical coefficients in held-out scores.
    x, y, w = [], [], []
    for mid, r in rows:
        keep = r["shape_y"] >= 0
        if not keep.any():
            raise ValueError("empty shape training match")
        x.append(r["shape_x"][keep])
        y.append(r["shape_y"][keep])
        w.append(np.full(keep.sum(), 1. / (keep.sum() * len(sources) * count[provider(mid)])))
    w = np.concatenate(w)
    w *= len(w) / w.sum()
    shape = LogisticRegression(C=1., fit_intercept=False, solver="lbfgs", max_iter=1000,
                               random_state=0).fit(np.concatenate(x), np.concatenate(y), sample_weight=w)
    result = copy.deepcopy(cfg)
    result["evidence"].update({"shape_" + k: float(v) for k, v in zip(SHAPE_KEYS, shape.coef_[0])})
    return model, result


def metrics(gt, codes) -> dict:
    result = pm.evaluate(gt, codes, bootstrap_n=0)
    return {"spell": float(result["spell_f1_mean"]), "macro": float(result["macro_f1"]),
            "dead_f1": float(result["per_class"]["D"]["f1"]),
            "possessions": int(result["n_pred_spells"])}


def score_fold(mid, model, cfg, seeds, recipes):
    ext = load_match(mid)
    rows = []
    for scenario, seed in conditions(seeds):
        match = ext.match if seed is None else perturb(ext.match, provider(mid), scenario, seed)
        feats = features.compute(match)
        original = evidence.scores(feats, cfg["evidence"])
        inclusive = [(a, b - 1) for a, b in rd.blocks(match)]
        baseline = metrics(ext.codes, decode_blocks(original, inclusive, cfg, feats))
        activity = model.predict(rd.player_design(match))
        use = match.graded_mask()
        noise = dict(ball_missing=float(np.mean(~np.isfinite(match.ball_xy[use]).all(axis=1))),
                     players_seen=float(np.mean(np.isfinite(match.players_xy[use]).all(axis=2).sum(axis=1))),
                     mean_speed=float(np.nanmean(match.players_speed[use])))
        for recipe in recipes:
            proposed = rd.compose_scores(feats, cfg["evidence"], activity, rd.RECIPES[recipe])
            if not np.allclose(proposed[:, 0] - proposed[:, 1], original[:, 0] - original[:, 1], atol=1e-6):
                raise ValueError("ownership evidence changed")
            pred = decode_blocks(proposed, inclusive, cfg, feats)
            candidate = metrics(ext.codes, pred)
            rows.append(dict(match=mid, provider=provider(mid), scenario=scenario, seed=seed,
                             recipe=recipe, baseline=baseline, candidate=candidate, noise=noise))
    return rows


def validate_rows(rows, ids, seeds, recipe):
    expected = {(mid, s, seed) for mid in ids for s, seed in conditions(seeds)}
    found = [(r["match"], r["scenario"], r["seed"]) for r in rows]
    if len(set(found)) != len(found) or set(found) != expected or any(r["recipe"] != recipe for r in rows):
        raise ValueError("incomplete, duplicate or mixed evaluation rows")
    for r in rows:
        for arm in ("baseline", "candidate"):
            if any(not np.isfinite(r[arm][k]) or not 0 <= r[arm][k] <= 1 for k in ("spell", "macro", "dead_f1")):
                raise ValueError("invalid aggregate metrics")


def summarize(rows) -> dict:
    ids = sorted({r["match"] for r in rows})
    if not ids:
        raise ValueError("no evaluation rows")
    groups = {}
    for scenario in ("clean", *SCENARIOS):
        per_match = []
        for mid in ids:
            cases = [r for r in rows if r["match"] == mid and r["scenario"] == scenario]
            if not cases:
                raise ValueError("missing match/scenario")
            values = {arm: {k: float(np.mean([r[arm][k] for r in cases]))
                            for k in ("spell", "macro", "dead_f1")} for arm in ("baseline", "candidate")}
            delta = {k: values["candidate"][k] - values["baseline"][k] for k in values["baseline"]}
            per_match.append(dict(match=mid, provider=provider(mid), **values, delta=delta))
        groups[scenario] = dict(matches=per_match,
            delta={k: float(np.mean([r["delta"][k] for r in per_match])) for k in ("spell", "macro", "dead_f1")})
    d = np.array([r["delta"]["spell"] for r in groups["standard"]["matches"]])
    rng = np.random.RandomState(42017)
    interval = np.percentile(d[rng.randint(len(d), size=(20000, len(d)))].mean(axis=1), [2.5, 97.5]).tolist()
    sign_flip = [np.mean(d * signs) for signs in itertools.product((-1, 1), repeat=len(d))]
    p = float(np.mean(np.abs(sign_flip) >= abs(d.mean()) - 1e-14))
    provider_gain = {s: float(np.mean([r["delta"]["spell"] for r in groups["standard"]["matches"]
                                     if r["provider"] == s])) for s in {provider(mid) for mid in ids}}
    passed = (d.mean() >= .03 and interval[0] > 0 and min(provider_gain.values()) > 0
        and groups["clean"]["delta"]["spell"] >= -.01
        and min(r["delta"]["spell"] for s in ("clean", "standard") for r in groups[s]["matches"]) >= -.02
        and min(groups[s]["delta"]["macro"] for s in ("clean", "standard")) >= -.01
        and np.mean([groups[s]["delta"]["spell"] for s in SCENARIOS]) >= .03
        and min(groups[s]["delta"]["spell"] for s in SCENARIOS) >= -.01)
    return dict(scenarios=groups, standard_delta=float(d.mean()), provider_gain=provider_gain,
                match_ci95=interval, sign_flip_p=p, numerical_gate=bool(passed),
                limitation="Historically reused matches; conditional evidence, not guaranteed target transfer.")


def select(summaries: dict) -> str | None:
    passed = [r for r in rd.RECIPES if summaries[r]["numerical_gate"]]
    return max(passed, key=lambda r: (summaries[r]["standard_delta"], r == "half")) if passed else None


def run_experiment():
    if not os.environ.get("RUNPOD_POD_ID"):
        raise RuntimeError("real-match training/evaluation requires bounded RunPod compute")
    signature = fingerprint()
    root = work_dir() / "br2" / signature[:16]
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(DEFAULT_CONFIG)
    rows = []
    for mid in DEV:
        start = time.monotonic()
        train = tuple(x for x in DEV if x != mid)
        model, fold_cfg = fit_fold(train, (mid,), cfg, root)
        fold_rows = score_fold(mid, model, fold_cfg, DEV_SEEDS, tuple(rd.RECIPES))
        rows.extend(fold_rows)
        print(json.dumps(dict(stage="development", evaluation_match=mid, train_ids=train,
                              seconds=round(time.monotonic() - start, 2), rows=fold_rows)), flush=True)
    if signature != fingerprint():
        raise ValueError("source changed during evaluation")
    summaries = {}
    for recipe in rd.RECIPES:
        relevant = [r for r in rows if r["recipe"] == recipe]
        validate_rows(relevant, DEV, DEV_SEEDS, recipe)
        summaries[recipe] = summarize(relevant)
    selected = select(summaries)
    report = dict(source_sha256=signature, development=summaries, selected=selected,
                  production_promotion=False, target_evaluated=False)
    print(json.dumps(dict(stage="development_summary", **report)), flush=True)
    if selected:
        # Candidate identity is irrevocably recorded before any confirmation results.
        (root / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
        model, fold_cfg = fit_fold(DEV, CONFIRM, cfg, root)
        confirm_rows = [r for mid in CONFIRM for r in score_fold(mid, model, fold_cfg, CONFIRM_SEEDS, (selected,))]
        validate_rows(confirm_rows, CONFIRM, CONFIRM_SEEDS, selected)
        report["confirmation"] = summarize(confirm_rows)
        transfer = {}
        for source in ("idsse", "metrica"):
            train = tuple(mid for mid in DEV if provider(mid) == source)
            test = tuple(mid for mid in DEV if provider(mid) != source)
            model, fold_cfg = fit_fold(train, test, cfg, root)
            transfer_rows = [r for mid in test for r in score_fold(mid, model, fold_cfg, DEV_SEEDS, (selected,))]
            validate_rows(transfer_rows, test, DEV_SEEDS, selected)
            transfer[source] = summarize(transfer_rows)
        report["source_excluded"] = transfer
        report["external_gate"] = report["confirmation"]["numerical_gate"] and all(
            s["standard_delta"] > 0 and s["scenarios"]["clean"]["delta"]["spell"] >= -.01
            for s in transfer.values())
    else:
        report["external_gate"] = False
    (root / "aggregate_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(stage="final_summary", **report)), flush=True)
    return report


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter("ignore", RuntimeWarning)
        # The reviewed functions emit only aggregate JSON. Third-party errors stay at this boundary.
        run_experiment()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(json.dumps(dict(error=type(exc).__name__, locations=[
            dict(file=Path(f.filename).name, function=f.name, line=f.lineno)
            for f in traceback.extract_tb(exc.__traceback__)])), flush=True)
        raise SystemExit(1) from None
