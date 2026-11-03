"""Spell-edit verifier (W6-M2): learned acceptance of decodes proposed at other switch costs.

The production Viterbi path is one point of a family: decoding the same evidence with every
switch cost scaled by SCALES gives paths with more (scale < 1) or fewer (> 1) spells. An *edit*
is a maximal stretch where one scaled path differs from the base path, carrying the scaled
path's labels there. A gradient-boosted regressor, trained on external matches only
(scripts/train_v6.py), predicts each edit's effect on spellF1 from spell-level features
(duration, evidence inside and around it, touches by each team, ball and player context); edits
with a predicted gain above THRESHOLD are applied greedily, best first, never overlapping or
touching an applied one. The verifier can only choose among paths the decoder itself produces.

Team quantities are signed towards the edit's reference team, so swapping home and away swaps
the decision. Pure numpy; sklearn only to fit/apply the regressor. Deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from possession import decoder

SCALES = (0.35, 0.5, 0.7, 1.4, 2.0)
CONTEXT_FRAMES = (25, 75)
THRESHOLD = 0.0
IOUS = np.array([0.3, 0.4, 0.5, 0.6, 0.7])   # evaluation/possession_metrics.SPELL_IOUS (tested)
HGB_PARAMS = dict(loss="squared_error", learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                  min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
                  random_state=0)
FEATURE_KEYS = ("touch_vote", "owner_sign", "control_balance", "ball_present", "nearest_home_m",
                "nearest_away_m", "ball_speed_m_s", "ball_x_norm", "ball_y_norm",
                "all_mean_speed_smooth_m_s", "frac_players_slow", "ball_moving_frac",
                "ball_out_frac", "ball_slow_far_frac")
FEATURE_NAMES = (
    "log_dur", "n_sub", "new_ref", "new_other", "new_dead", "old_ref", "old_other", "old_dead",
    "scale_min", "scale_max", "n_scales", "z_mean", "z_max", "z_min", "gain_sum", "dead_mean",
    "touch_ref", "touch_other", "owner_mean", "control_mean", "ball_seen", "near_ref",
    "near_other", "ball_speed_max", "ball_speed_mean", "ball_speed_std", "ball_centre_x",
    "ball_centre_y", "players_speed", "players_slow", "ball_moving", "ball_out", "slow_far",
    "prod_run_a", "prod_run_b", "prod_left", "prod_right", "left_label_ref", "left_label_dead",
    "right_label_ref", "right_label_dead") + tuple(
    f"{k}{w}" for w in CONTEXT_FRAMES for k in ("z_before", "z_after", "dead_before", "dead_after",
                                         "touch_ref_ctx", "touch_other_ctx"))


def viterbi_decode(decoder_config: dict):
    """decode_fn(evidence, cost_scale) for the production Viterbi of config["decoder"], every
    switch cost multiplied by cost_scale."""
    options = dict(decoder_config)
    cost = decoder.switch_matrix(**options.pop("switch_cost"))
    if float(options.pop("restart_discount", 0.0)) > 0.0:
        raise ValueError("the verifier does not support decoder.restart_discount")

    def decode_fn(emis, scale):
        return decoder.decode(np.asarray(emis, dtype=np.float64), cost * scale, **options)
    return decode_fn


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def propose(emis, decode_fn, scales=SCALES):
    """(base path, edits) for one block; an edit is {a, b, new, scales}."""
    prod = np.asarray(decode_fn(emis, 1.0), dtype=np.int8)
    found = {}
    for s in scales:
        alt = np.asarray(decode_fn(emis, s), dtype=np.int8)
        for a, b in _runs(alt != prod):
            key = (int(a), int(b), alt[a:b].tobytes())
            if key in found:
                found[key]["scales"].append(s)
            else:
                found[key] = {"a": int(a), "b": int(b), "new": alt[a:b].copy(), "scales": [s]}
    return prod, sorted(found.values(), key=lambda e: (e["a"], e["b"], min(e["scales"])))


def _mean(x):
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else np.nan


def _run_len_at(path, t):
    lab = path[t]
    a = t
    while a > 0 and path[a - 1] == lab:
        a -= 1
    b = t
    while b + 1 < len(path) and path[b + 1] == lab:
        b += 1
    return b - a + 1


def features(emis, feats, prod, edits, offset=0) -> np.ndarray:
    """(n_edits, len(FEATURE_NAMES)). feats is the full-match dict, offset the block's first frame.
    Reference team: majority team of the edit's new team frames (or of the replaced frames when
    the edit is all dead); ties go to the team the summed team logit favours."""
    e = np.asarray(emis, dtype=np.float64)
    z = e[:, 0] - e[:, 1]
    dead = e[:, 2] - np.logaddexp(e[:, 0], e[:, 1])
    n = len(e)
    F = {k: np.asarray(feats[k], dtype=np.float64)[offset:offset + n] for k in FEATURE_KEYS}
    out = np.empty((len(edits), len(FEATURE_NAMES)), dtype=np.float64)
    for i, ed in enumerate(edits):
        a, b, new = ed["a"], ed["b"], ed["new"]
        old = prod[a:b]
        content = new if (new < 2).any() else old
        n_home, n_away = int(np.sum(content == 0)), int(np.sum(content == 1))
        ref = (0 if n_home > n_away else 1) if n_home != n_away else (0 if z[a:b].sum() >= 0 else 1)
        s = 1.0 if ref == 0 else -1.0
        other = 1 - ref
        speed = F["ball_speed_m_s"][a:b]
        near_ref = F["nearest_home_m"] if ref == 0 else F["nearest_away_m"]
        near_other = F["nearest_away_m"] if ref == 0 else F["nearest_home_m"]
        r = {
            "log_dur": np.log(b - a), "n_sub": float(len(_runs(new[1:] != new[:-1])) + 1),
            "new_ref": np.mean(new == ref), "new_other": np.mean(new == other),
            "new_dead": np.mean(new == 2), "old_ref": np.mean(old == ref),
            "old_other": np.mean(old == other), "old_dead": np.mean(old == 2),
            "scale_min": min(ed["scales"]), "scale_max": max(ed["scales"]),
            "n_scales": len(ed["scales"]),
            "z_mean": s * z[a:b].mean(), "z_max": (s * z[a:b]).max(), "z_min": (s * z[a:b]).min(),
            "gain_sum": float(e[np.arange(a, b), new].sum() - e[np.arange(a, b), old].sum()),
            "dead_mean": dead[a:b].mean(),
            "touch_ref": float(np.sum(s * F["touch_vote"][a:b] > 0)),
            "touch_other": float(np.sum(s * F["touch_vote"][a:b] < 0)),
            "owner_mean": s * _mean(F["owner_sign"][a:b]),
            "control_mean": s * _mean(F["control_balance"][a:b]),
            "ball_seen": _mean(F["ball_present"][a:b]),
            "near_ref": _mean(near_ref[a:b]), "near_other": _mean(near_other[a:b]),
            "ball_speed_max": np.nanmax(speed) if np.isfinite(speed).any() else np.nan,
            "ball_speed_mean": _mean(speed),
            "ball_speed_std": float(np.nanstd(speed)) if np.isfinite(speed).sum() > 1 else np.nan,
            "ball_centre_x": abs(_mean(F["ball_x_norm"][a:b]) - 0.5),
            "ball_centre_y": abs(_mean(F["ball_y_norm"][a:b]) - 0.5),
            "players_speed": _mean(F["all_mean_speed_smooth_m_s"][a:b]),
            "players_slow": _mean(F["frac_players_slow"][a:b]),
            "ball_moving": _mean(F["ball_moving_frac"][a:b]),
            "ball_out": _mean(F["ball_out_frac"][a:b]),
            "slow_far": _mean(F["ball_slow_far_frac"][a:b]),
            "prod_run_a": np.log(_run_len_at(prod, a)), "prod_run_b": np.log(_run_len_at(prod, b - 1)),
            "prod_left": np.log(_run_len_at(prod, a - 1)) if a > 0 else np.nan,
            "prod_right": np.log(_run_len_at(prod, b)) if b < n else np.nan,
            "left_label_ref": float(prod[a - 1] == ref) if a > 0 else np.nan,
            "left_label_dead": float(prod[a - 1] == 2) if a > 0 else np.nan,
            "right_label_ref": float(prod[b] == ref) if b < n else np.nan,
            "right_label_dead": float(prod[b] == 2) if b < n else np.nan,
        }
        for w in CONTEXT_FRAMES:
            lo, hi = slice(max(a - w, 0), a), slice(b, min(b + w, n))
            tv = F["touch_vote"]
            r[f"z_before{w}"] = s * z[lo].mean() if a > 0 else np.nan
            r[f"z_after{w}"] = s * z[hi].mean() if b < n else np.nan
            r[f"dead_before{w}"] = dead[lo].mean() if a > 0 else np.nan
            r[f"dead_after{w}"] = dead[hi].mean() if b < n else np.nan
            r[f"touch_ref_ctx{w}"] = float(np.sum(s * tv[lo] > 0) + np.sum(s * tv[hi] > 0))
            r[f"touch_other_ctx{w}"] = float(np.sum(s * tv[lo] < 0) + np.sum(s * tv[hi] < 0))
        out[i] = [r[k] for k in FEATURE_NAMES]
    return out


def accept(prod, edits, score, threshold=THRESHOLD):
    """Apply edits greedily by descending score above threshold, skipping any edit that
    overlaps or touches an applied one."""
    path = prod.copy()
    taken = np.zeros(len(prod) + 1, dtype=bool)
    for i in np.argsort(-np.asarray(score), kind="stable"):
        if score[i] <= threshold:
            break
        a, b = edits[i]["a"], edits[i]["b"]
        if taken[max(a - 1, 0):min(b + 1, len(prod))].any():
            continue
        path[a:b] = edits[i]["new"]
        taken[a:b] = True
    return path


def block_table(emis, feats, offset, decode_fn):
    """(base path, edits, feature matrix) for one block."""
    emis = np.asarray(emis, dtype=np.float64)
    prod, edits = propose(emis, decode_fn)
    return prod, edits, features(emis, feats, prod, edits, offset)


@dataclass
class VerifierModel:
    regressor: HistGradientBoostingRegressor
    feature_names: tuple = FEATURE_NAMES
    threshold: float = THRESHOLD


def decode_block(model: VerifierModel, emis, feats, offset, decode_fn) -> np.ndarray:
    """Verified codes for one block (base path + accepted edits)."""
    if tuple(model.feature_names) != FEATURE_NAMES:
        raise ValueError("verifier model was trained on a different feature layout")
    prod, edits, X = block_table(emis, feats, offset, decode_fn)
    if not edits:
        return prod
    return accept(prod, edits, model.regressor.predict(X), model.threshold)


# --- training side (external labels only) -------------------------------------------------

def _spells(codes):
    codes = np.asarray(codes)
    if not len(codes):
        e = np.empty(0, dtype=np.int64)
        return e, e, e
    br = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    st, en = np.concatenate(([0], br)), np.concatenate((br, [len(codes)]))
    keep = codes[st] < 2
    return st[keep], en[keep], codes[st][keep].astype(np.int64)


def true_positives(gt, pred) -> np.ndarray:
    """Spell true positives at each IoU of IOUS: the official one-to-one greedy match
    (possession_metrics.spell_scores) computed in one pass over IoU-sorted overlaps."""
    gs, ge, gc = gt
    ps, pe, pc = pred
    tp = np.zeros(len(IOUS), dtype=np.int64)
    if not len(gs) or not len(ps):
        return tp
    lo = np.searchsorted(pe, gs, side="right")
    hi = np.searchsorted(ps, ge, side="left")
    items = []
    for i in range(len(gs)):
        for j in range(lo[i], hi[i]):
            if pc[j] != gc[i]:
                continue
            inter = min(ge[i], pe[j]) - max(gs[i], ps[j])
            if inter > 0:
                items.append((-(inter / float(max(ge[i], pe[j]) - min(gs[i], ps[j]))), i, j))
    items.sort()
    used_g, used_p, count, k = set(), set(), 0, len(IOUS) - 1
    for neg, i, j in items:
        while k >= 0 and -neg < IOUS[k]:
            tp[k] = count
            k -= 1
        if k < 0:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        count += 1
    while k >= 0:
        tp[k] = count
        k -= 1
    return tp


def labels(gt_block, prod, edits) -> np.ndarray:
    """Training target: change of the linearised spellF1 utility U = mean_k TP_k - lambda * P
    (lambda = mean_k F1_k / 2, the first-order term of mean_k 2 TP_k / (P + G)) when the edit
    alone is applied to the base path."""
    gt = _spells(gt_block)
    sp0 = _spells(prod)
    tp0, p0 = true_positives(gt, sp0).astype(np.float64), len(sp0[0])
    lam = float(np.mean(2 * tp0 / max(p0 + len(gt[0]), 1))) / 2
    out = np.empty(len(edits))
    for i, ed in enumerate(edits):
        path = prod.copy()
        path[ed["a"]:ed["b"]] = ed["new"]
        sp = _spells(path)
        out[i] = float(np.mean(true_positives(gt, sp) - tp0)) - lam * (len(sp[0]) - p0)
    return out


def fit(tables) -> VerifierModel:
    """tables: [(X, y)] per training block."""
    X = np.concatenate([t[0] for t in tables])
    y = np.concatenate([t[1] for t in tables])
    return VerifierModel(HistGradientBoostingRegressor(**HGB_PARAMS).fit(X, y))
