"""Synthetic-only invariants for BR2; no match assets or trained artifacts."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.special import expit

from possession import evidence, features, robust_dead as rd
from possession.io import Match


def fixture(n=1000):
    t = np.arange(n) / 25.
    xy = np.zeros((n, 4, 2), np.float32)
    xy[:, :, 0] = (np.array([20., 30., 70., 80.])[None, :] + 2 * np.sin(t[:, None])) / 105.
    xy[:, :, 1] = np.array([.2, .4, .6, .8])
    return Match(xy, np.full((n, 4), 2., np.float32), xy[:, 0].copy(),
                 np.full(n, 2., np.float32), np.array([0, 0, 1, 1], np.int8),
                 [(1, 0, n // 2 - 1), (2, n // 2, n - 1)], (105., 68.))


class RobustActivityTest(unittest.TestCase):
    def test_shape_and_non_mutation(self):
        m = fixture()
        before = m.players_xy.copy()
        x = rd.player_design(m)
        self.assertEqual(x.shape, (m.n_frames, 25))
        self.assertTrue(np.isfinite(x[100:400]).all())
        self.assertTrue(np.isfinite(x[600:900]).all())
        np.testing.assert_array_equal(m.players_xy, before)

    def test_ball_and_team_independence(self):
        m = fixture()
        other = replace(m, ball_xy=np.full_like(m.ball_xy, np.nan), team_of_slot=1 - m.team_of_slot,
                        players_speed=np.full_like(m.players_speed, 100.))
        np.testing.assert_array_equal(rd.player_design(m), rd.player_design(other))

    def test_half_independence(self):
        m = fixture()
        baseline = rd.player_design(m)
        xy = m.players_xy.copy()
        xy[:500] = np.nan
        changed = rd.player_design(replace(m, players_xy=xy))
        np.testing.assert_array_equal(baseline[500:], changed[500:])

    def test_slot_permutation_and_pitch_units(self):
        m = fixture()
        permutation = [3, 1, 0, 2]
        other = replace(m, players_xy=m.players_xy[:, permutation])
        np.testing.assert_array_equal(rd.player_design(m), rd.player_design(other))
        other = replace(m, players_xy=m.players_xy / 2., pitch_m=(210., 136.))
        np.testing.assert_array_equal(rd.player_design(m), rd.player_design(other))

    def test_missing_middle_invalidates_displacement(self):
        xy = np.zeros((100, 1, 2))
        xy[50] = np.nan
        speed = rd.displacement_speed(xy, 12)
        self.assertTrue(np.isnan(speed[38:63]).all())
        self.assertTrue(np.isfinite(speed[20:30]).all())

    def test_reject_identity_jump(self):
        xy = np.zeros((100, 1, 2))
        xy[50:, :, 0] = 80.
        self.assertTrue(np.isnan(rd.displacement_speed(xy, 12)[38:62]).all())

    def test_empty_short_stationary_and_outside_halves(self):
        m = fixture(20)
        m.players_xy[:] = np.nan
        x = rd.player_design(m)
        self.assertFalse(np.isinf(x).any())
        self.assertTrue(np.isnan(x[:, :20]).all())
        m = fixture()
        m.players_xy[:] = .5
        m.halves = [(1, 100, 400)]
        x = rd.player_design(m)
        self.assertTrue(np.isnan(x[:100]).all())
        self.assertTrue(np.isfinite(x[150:350]).all())
        np.testing.assert_array_equal(x[150:350, :10], 0.)

    def test_directional_windows(self):
        x = np.array([1., 2., 3., 4.])
        np.testing.assert_allclose(rd.window_mean(x, 2, "past"), [np.nan, 1., 1.5, 2.5], equal_nan=True)
        np.testing.assert_allclose(rd.window_mean(x, 2, "next"), [2.5, 3.5, 4., np.nan], equal_nan=True)

    def test_invalid_half_bounds(self):
        m = fixture()
        m.halves = [(1, 0, 500), (2, 500, 999)]
        with self.assertRaises(ValueError):
            rd.player_design(m)

    def test_ownership_preservation_and_zero_blend(self):
        f = features.compute(fixture())
        params = dict(evidence.DEFAULT_PARAMS)
        orig = evidence.scores(f, params)
        changed = rd.compose_scores(f, params, np.zeros(len(orig)), .5)
        np.testing.assert_allclose(changed[:, 0] - changed[:, 1], orig[:, 0] - orig[:, 1], atol=1e-6)
        np.testing.assert_array_equal(rd.compose_scores(f, params, np.zeros(len(orig)), 0), orig)
        np.testing.assert_allclose(rd.rule_offset(f, params) + rd.activity_logit(f, params),
                                   evidence.dead_logit(f, params), atol=1e-6)

    def test_fit_determinism_and_missing_features(self):
        rng = np.random.RandomState(2)
        x = rng.normal(size=(800, 3))
        offset = rng.normal(size=800)
        y = (rng.rand(800) < expit(x[:, 0] + offset)).astype(float)
        x[:30, 1] = np.nan
        x[:, 2] = np.nan
        a = rd.fit_activity(x, y, offset, np.ones(800))
        b = rd.fit_activity(x, y, offset, np.ones(800))
        np.testing.assert_array_equal(a.beta, b.beta)
        self.assertTrue(np.isfinite(a.predict(x)).all())
        self.assertGreater(a.beta[1], .5)
        before = a.mean.copy()
        a.predict(np.full((5, 3), 999.))
        np.testing.assert_array_equal(before, a.mean)

    def test_training_boundaries(self):
        x = np.ones((20, 2))
        for y, w in [(np.ones(20), np.ones(20)), (np.arange(20) % 2, np.zeros(20))]:
            with self.assertRaises(ValueError):
                rd.fit_activity(x, y, np.zeros(20), w)


if __name__ == "__main__":
    unittest.main()
