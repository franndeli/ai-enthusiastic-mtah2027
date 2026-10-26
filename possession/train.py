"""Leave-one-match-out training and validation on DFL (ROUTE_A_PLAN.md T4). Aggregates only.

    scripts/py -m possession.train                       # LOMO table: baseline vs argmax vs Viterbi
    scripts/py -m possession.train --fit-all                       # final model -> work dir

Model selection happens here, on external matches scored with the official spellF1, never on the
public 5 minutes (EVALUATION.md misranking trap). Every table gets the same seeded ATD-like ball
gaps (see drop_ball), because the target ATD misses the ball on ~41% of play frames and DFL never does.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from possession.features import features
from possession.frames import AWAY, HOME, A, D, H, X, FrameTable, period_runs
from possession.model import Model, argmax_decode, decode, fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm  # noqa: E402
from starter.baseline import smooth  # noqa: E402

SEED = 2027
# ATD-like ball gaps, calibrated by possession.domain_gap (docs/research/domain_gap.md).
DROPOUT_RATE = 0.407       # ATD ball missing share on play frames
GAP_MEDIAN_S = 10.7        # ATD gaps are long and heavy-tailed: lognormal fits median/mean/p90
GAP_SIGMA = 0.97
DEAD_RATIO = 5.0           # public window: P(missing | D) / P(missing | alive) = 0.336 / 0.065
TRAIN_STEP_S = 0.2         # subsample training rows; neighbours are near-duplicates
EMISSION_SCALES = (0.2, 0.5, 1.0, 2.0, 4.0)
assert (HOME, AWAY) == (H, A)  # nearest_team_codes and smooth() treat a team code as its label


def work_dir() -> Path:
    return Path(os.environ.get("MTAH_WORK_DIR", Path.home() / ".mtah2027" / "work"))


def drop_ball(table: FrameTable, rng: np.random.RandomState, rate: float = DROPOUT_RATE,
              gap_median_s: float = GAP_MEDIAN_S, gap_sigma: float = GAP_SIGMA,
              dead_ratio: float = DEAD_RATIO) -> FrameTable:
    """Copy with the ball hidden in ATD-like gaps covering ~rate of the frames.

    Gap lengths are lognormal; gaps start dead_ratio times more often while the ball is dead,
    because on the ATD a missing ball is itself evidence of a stoppage.
    """
    n = table.n_frames
    mean_len = gap_median_s * np.exp(gap_sigma ** 2 / 2) * table.fps
    dead = table.labels == D
    d_share = dead[table.period > 0].mean()
    p_dead = min(0.95, rate / (d_share + (1 - d_share) / dead_ratio))
    p_alive = p_dead / dead_ratio
    # Renewal approximation: missing share p needs a start hazard of p / ((1 - p) * mean length).
    hazard = np.where(dead, p_dead / ((1 - p_dead) * mean_len), p_alive / ((1 - p_alive) * mean_len))
    starts = rng.rand(n) < hazard
    lengths = np.maximum(1, np.round(rng.lognormal(np.log(gap_median_s * table.fps), gap_sigma, n)))
    mask = np.zeros(n, dtype=bool)
    pos = 0
    for start in np.flatnonzero(starts):
        if start >= pos:
            pos = start + int(lengths[start])
            mask[start:pos] = True
    ball = table.ball_xy_m.copy()
    ball[mask] = np.nan
    return FrameTable(table.fps, table.pitch_m, table.period, ball, table.players_xy_m,
                      table.slot_team, table.labels)


def mirror(values: np.ndarray, labels: np.ndarray, antisymmetric: np.ndarray) -> tuple:
    """Team-swapped copy: antisymmetric columns negate, H and A trade places."""
    flipped = values.copy()
    flipped[:, antisymmetric] *= -1
    swapped = np.where(labels == H, A, np.where(labels == A, H, labels)).astype(labels.dtype)
    return flipped, swapped


def nearest_team_codes(table: FrameTable) -> np.ndarray:
    """starter/baseline.py step 1 on a FrameTable: team of the nearest player, -1 if no ball."""
    dist = np.hypot(table.players_xy_m[..., 0] - table.ball_xy_m[:, None, 0],
                    table.players_xy_m[..., 1] - table.ball_xy_m[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = dist.argmin(axis=1)
    codes = table.slot_team[nearest].astype(np.int8)
    codes[~np.isfinite(dist.min(axis=1))] = -1
    return codes


def baseline_codes(table: FrameTable, window: int = 50) -> np.ndarray:
    return pm.label_codes(smooth(nearest_team_codes(table), window))


def prepare(table: FrameTable, rng: np.random.RandomState, feature_fn=features) -> dict:
    """Dropout-augmented table, its features, and the subsampled + mirrored training rows."""
    dropped = drop_ball(table, rng)
    feats = feature_fn(dropped)
    step = max(1, int(round(TRAIN_STEP_S * table.fps)))
    rows = np.flatnonzero((table.period > 0) & (table.labels != X))[::step]
    values, labels = feats.values[rows], table.labels[rows]
    flipped, swapped = mirror(values, labels, feats.antisymmetric)
    runs = [table.labels[a:b] for a, b in period_runs(table.period)]
    mirrored_runs = [np.where(r == H, A, np.where(r == A, H, r)) for r in runs]
    return dict(table=dropped, values=feats.values, train_x=np.vstack([values, flipped]),
                train_y=np.concatenate([labels, swapped]), runs=runs + mirrored_runs)


def train(prepared: list, emission_scale: float) -> Model:
    return fit(np.vstack([p["train_x"] for p in prepared]),
               np.concatenate([p["train_y"] for p in prepared]),
               [r for p in prepared for r in p["runs"]], emission_scale)


def score(labels: np.ndarray, pred: np.ndarray) -> tuple:
    result = pm.evaluate(labels.astype(np.int8), pred.astype(np.int8))
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]


def load_tables() -> list:
    paths = sorted((work_dir() / "dfl").glob("*.npz"))
    if not paths:
        sys.exit("no DFL tables found; run scripts/fetch_external.py and possession.io_dfl first")
    return [(p.stem, FrameTable.load(p)) for p in paths]


LOMO_METRICS = ("spell_f1", "macro_f1", "n_pred", "n_gt", "team_acc_seen", "team_acc_missing")


def lomo(tables: list, feature_fn=features, train_fn=train, scales=EMISSION_SCALES) -> dict:
    """Print the leave-one-match-out table; return {method: {metric: mean over matches}}.

    feature_fn and train_fn let experiments swap one component while every other step (dropout,
    seeds, scoring) stays identical, so results are paired with the committed pipeline.
    team_acc_* is H/A accuracy on frames where truth and prediction are both alive, split by
    whether the (dropped-out) ball is seen.
    """
    rng = np.random.RandomState(SEED)
    prepared = [prepare(t, rng, feature_fn) for _, t in tables]
    methods = ["base50", "argmax"] + [f"vit{s}" for s in scales]
    print("spellF1 (macroF1) [predicted spells] per held-out match")
    print("  ".join(f"{h:>22}" for h in ["match", "gt_spells"] + methods))
    sums = {m: np.zeros(len(LOMO_METRICS)) for m in methods}
    for i, (name, table) in enumerate(tables):
        start = time.time()
        model = train_fn(prepared[:i] + prepared[i + 1:], scales[0])
        held = prepared[i]
        seen = ~np.isnan(held["table"].ball_xy_m[:, 0])
        preds = [baseline_codes(held["table"]),
                 argmax_decode(model, held["values"], table.period)]
        preds += [decode(model.with_emission_scale(s), held["values"], table.period)
                  for s in scales]
        cells, n_gt = [], 0
        for method, pred in zip(methods, preds):
            spell, macro, n_pred, n_gt = score(table.labels, pred)
            alive = (table.labels < D) & (pred < D)
            correct = pred == table.labels
            team = [correct[alive & s].sum() / max((alive & s).sum(), 1) for s in (seen, ~seen)]
            sums[method] += (spell, macro, n_pred, n_gt, *team)
            cells.append(f"{spell:.4f} ({macro:.3f}) [{n_pred}]")
        print("  ".join(f"{c:>22}" for c in [name, str(n_gt)] + cells),
              f"  {time.time() - start:.0f}s")
    means = {m: dict(zip(LOMO_METRICS, s / len(tables))) for m, s in sums.items()}
    rows = [("MEAN", "{spell_f1:.4f} ({macro_f1:.3f})"), ("spells", "{n_pred:.0f} of {n_gt:.0f}"),
            ("team seen/missing", "{team_acc_seen:.3f} / {team_acc_missing:.3f}")]
    for label, fmt in rows:
        print("  ".join(f"{c:>22}" for c in [label, ""] + [fmt.format(**means[m]) for m in methods]))
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fit-all", action="store_true", help="fit on every match and save")
    parser.add_argument("--emission-scale", type=float, default=0.5)  # docs/research/domain_gap.md
    args = parser.parse_args()
    tables = load_tables()
    print(f"{len(tables)} DFL matches, {sum(t.n_frames for _, t in tables)} frames")
    if not args.fit_all:
        lomo(tables)
        return
    rng = np.random.RandomState(SEED)
    model = train([prepare(t, rng) for _, t in tables], args.emission_scale)
    path = work_dir() / "model.npz"
    model.save(path)
    print(f"model saved to the work dir as {path.name} (emission_scale {args.emission_scale})")


if __name__ == "__main__":
    main()
