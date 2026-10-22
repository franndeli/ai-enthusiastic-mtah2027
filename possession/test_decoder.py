"""Synthetic decoder invariants; never reads match assets."""
import unittest
import numpy as np
from possession.decoder import decode, switch_matrix


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.cost = switch_matrix(6, 6, 6)
        self.e = np.tile([0., -4., -8.], (100, 1))

    def test_flicker_and_short_spell(self):
        self.e[40] = [-4, 0, -8]
        self.assertTrue(np.all(decode(self.e, self.cost) == 0))
        self.e[40:60] = [-4, 0, -8]
        for duration in (1, 2):
            path = decode(self.e, self.cost, min_duration_frames=duration)
            np.testing.assert_array_equal(path[40:60], np.ones(20))

    def test_duration_forbids_strong_single_frame(self):
        self.e[40] = [-100, 0, -100]
        path = decode(self.e, self.cost, min_duration_frames=2)
        bounds = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
        self.assertTrue(np.all(np.diff(bounds) >= 2))

    def test_scale_and_determinism(self):
        self.e[40:44] = [-2, 0, -8]
        self.assertFalse(np.any(decode(self.e, self.cost) == 1))
        scale = np.zeros(100)
        path = decode(self.e, self.cost, cost_scale=scale)
        self.assertTrue(np.all(path[40:44] == 1))
        np.testing.assert_array_equal(path, decode(self.e, self.cost, cost_scale=scale))
        matrix_scale = np.broadcast_to(scale[:, None, None], (100, 3, 3))
        np.testing.assert_array_equal(path, decode(self.e, self.cost, cost_scale=matrix_scale))

    def test_invalid(self):
        self.e[0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode(self.e, self.cost)
        with self.assertRaises(ValueError):
            decode(np.zeros((3, 3)), self.cost, cost_scale=np.ones(2))

    def test_empty_and_legacy_initial(self):
        self.assertEqual(len(decode(np.empty((0, 3)), self.cost)), 0)
        np.testing.assert_array_equal(decode(np.zeros((4, 3)), self.cost, [0, 1, 0]), [1]*4)

    def test_matches_exhaustive_paths(self):
        import itertools
        rng = np.random.RandomState(7)
        evidence = rng.normal(size=(6, 3))
        scale = rng.uniform(size=(6, 3, 3))
        cost = switch_matrix(.4, .7, .8)
        for duration in (1, 2, 3):
            best = -np.inf
            for sequence in itertools.product(range(3), repeat=6):
                bounds = np.r_[0, np.flatnonzero(np.diff(sequence)) + 1, 6]
                if np.any(np.diff(bounds) < duration):
                    continue
                score = sum(evidence[t, s] for t, s in enumerate(sequence))
                score -= sum(cost[sequence[t-1], sequence[t]] *
                             scale[t, sequence[t-1], sequence[t]] for t in range(1, 6))
                best = max(best, score)
            path = decode(evidence, cost, cost_scale=scale, min_duration_frames=duration)
            score = evidence[np.arange(6), path].sum()
            score -= sum(cost[path[t-1], path[t]] * scale[t, path[t-1], path[t]]
                         for t in range(1, 6))
            self.assertAlmostEqual(score, best)
