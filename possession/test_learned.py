"""Synthetic-input tests for possession.learned: shapes, determinism, team-swap symmetry."""
from __future__ import annotations

import unittest

import numpy as np

from possession import A, D, H, learned

N = 4000


def synthetic_features(seed: int = 0) -> tuple[dict, np.ndarray]:
    """Features with a planted signal: the nearer team owns the ball, a slow field means dead."""
    rng = np.random.RandomState(seed)
    owner = np.repeat(rng.randint(0, 3, N // 200 + 1), 200)[:N]       # H/A/D blocks of 8 s
    near_home = np.where(owner == H, 1.0, 8.0) + rng.rand(N)
    near_away = np.where(owner == A, 1.0, 8.0) + rng.rand(N)
    speed = np.where(owner == D, 0.5, 3.0) + 0.1 * rng.randn(N)
    sign = np.where(owner == H, 1.0, np.where(owner == A, -1.0, 0.0))
    gap = np.full(N, 1e6)
    feats = {
        "ball_present": np.ones(N), "ball_speed_m_s": 5 * rng.rand(N),
        "ball_out_frac": np.zeros(N), "frames_since_ball": np.zeros(N),
        "frames_until_ball": np.zeros(N), "frames_since_team": gap, "frames_until_team": gap,
        "all_mean_speed_m_s": speed, "all_mean_speed_smooth_m_s": speed,
        "frac_players_slow": (owner == D).astype(float),
        "nearest_home_m": near_home, "nearest_away_m": near_away,
        "home_spread_m": 10 + rng.rand(N), "away_spread_m": 10 + rng.rand(N),
        "home_mean_speed_m_s": speed, "away_mean_speed_m_s": speed,
        "home_n_tracked": np.full(N, 10.0), "away_n_tracked": np.full(N, 10.0),
        "control_balance": 0.8 * sign, "last_team_sign": sign, "next_team_sign": sign,
    }
    return {k: v.astype(np.float32) for k, v in feats.items()}, owner.astype(np.int8)


def small_model(feats: dict, owner: np.ndarray) -> learned.LearnedModel:
    x, y = learned.training_rows(feats, owner, stride=2)
    rows = {"a": (x, y), "b": (x, y)}
    return learned.fit_model(rows, max_iter=20)


class LearnedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.feats, cls.owner = synthetic_features()
        cls.model = small_model(cls.feats, cls.owner)

    def test_shape_dtype_finite(self):
        out = learned.scores(self.feats, model=self.model)
        self.assertEqual(out.shape, (N, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())

    def test_learns_planted_signal(self):
        out = learned.scores(self.feats, model=self.model)
        self.assertGreater(np.mean(out.argmax(axis=1) == self.owner), 0.9)

    def test_deterministic_given_model(self):
        a = learned.scores(self.feats, model=self.model)
        b = learned.scores(self.feats, model=self.model)
        np.testing.assert_array_equal(a, b)

    def test_refit_is_identical(self):
        again = small_model(self.feats, self.owner)
        np.testing.assert_array_equal(learned.scores(self.feats, model=self.model),
                                      learned.scores(self.feats, model=again))

    def test_team_swap_swaps_columns(self):
        out = learned.scores(self.feats, model=self.model)
        swapped = learned.scores(learned.swap_teams(self.feats), model=self.model)
        np.testing.assert_allclose(swapped[:, [1, 0, 2]], out, atol=1e-5)

    def test_nan_ball_features_are_accepted(self):
        feats = dict(self.feats)
        for name in ("nearest_home_m", "nearest_away_m", "ball_speed_m_s"):
            feats[name] = np.full(N, np.nan, dtype=np.float32)
        feats["ball_present"] = np.zeros(N, dtype=np.float32)
        self.assertTrue(np.isfinite(learned.scores(feats, model=self.model)).all())

    def test_weight_zero_is_rules_and_bad_weight_rejected(self):
        from possession import evidence
        np.testing.assert_allclose(learned.scores(self.feats, {"weight": 0.0}, model=self.model),
                                   evidence.scores(self.feats), atol=1e-6)
        with self.assertRaises(ValueError):
            learned.scores(self.feats, {"weight": 1.5}, model=self.model)

    def test_swap_labels_keeps_dead(self):
        np.testing.assert_array_equal(learned.swap_labels(np.array([H, A, D], dtype=np.int8)),
                                      [A, H, D])


if __name__ == "__main__":
    unittest.main()
