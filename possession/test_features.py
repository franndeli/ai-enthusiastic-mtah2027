import os
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import features as F  # noqa: E402
from possession.model import Model  # noqa: E402
from possession.tracking import Tracking  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def synthetic(n: int = 600, seed: int = 0) -> Tracking:
    """Four players walking about, the ball with the first home player and then the first away one,
    a stretch with no ball and a stretch with no player at all."""
    rng = np.random.RandomState(seed)
    start = np.array([[30.0, 20.0], [40.0, 50.0], [70.0, 30.0], [60.0, 45.0]])
    players = start[None] + np.cumsum(rng.normal(0, 0.05, (n, 4, 2)), axis=0)
    ball = np.where(np.arange(n)[:, None] < n // 2, players[:, 0], players[:, 2]) + 0.3
    ball[100:160] = np.nan
    players[300:320] = np.nan
    return Tracking(ball=ball.astype(np.float32), players=players.astype(np.float32),
                    team=np.array([0, 0, 1, 1], dtype=np.int8), fps=25.0, pitch=(105.0, 68.0),
                    blocks=[(0, n // 2), (n // 2, n)])


class TestFeatures(unittest.TestCase):
    def test_rolling_mean_side(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        past = F.rolling_mean_side(x, 2, future=False)
        nxt = F.rolling_mean_side(x, 2, future=True)
        self.assertTrue(np.isnan(past[0]) and np.isnan(nxt[-1]))
        np.testing.assert_allclose(past[1:], [1.0, 1.5, 2.5, 3.5])
        np.testing.assert_allclose(nxt[:-1], [2.5, 3.5, 4.5, 5.0])

    def test_compute_is_quiet_and_complete(self):
        trk = synthetic()
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # no "Mean of empty slice" and the like
            feats = F.compute(trk)
        for name, values in feats.items():
            self.assertEqual(values.shape, (trk.n,), name)
        for path in ("possession_v3.json", "possession_v4.json"):
            model = Model.from_json(os.path.join(CONFIG, path))
            dl, tl = model.dead_logit(feats), model.team_logit(feats)
            self.assertTrue(np.isfinite(dl).all() and np.isfinite(tl).all(), path)

    def test_v4_feature_sets_are_shipped(self):
        model = Model.from_json(os.path.join(CONFIG, "possession_v4.json"))
        self.assertEqual(tuple(model.p["dead"]["names"]), F.DEAD_FEATURES_V4)
        self.assertEqual(tuple(model.p["team"]["names"]), F.TEAM_FEATURES_V4)
        self.assertEqual(model.p["decode"], {"cost_dead": 40.0, "cost_team": 8.0})


if __name__ == "__main__":
    unittest.main()
