import importlib.util
import unittest

import numpy as np

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
if HAS_SCIPY:
    from possession.retrack import retrack_all


def _players(n=1500, seed=0):
    """22 smooth trajectories cut into ATD-like tracklets over 46 slots (23 per team)."""
    rng = np.random.RandomState(seed)
    t = np.arange(n) / 25.0
    true = np.zeros((n, 22, 2))
    for i in range(22):
        base = np.array([10 + 4 * (i % 11) + (50 if i >= 11 else 0), 5 + 5 * (i % 11) % 60])
        true[:, i] = base + np.stack([3 * np.sin(t / (3 + i % 5)), 2 * np.cos(t / (4 + i % 3))], 1)
    slots = np.full((n, 46, 2), np.nan)
    team = np.array([0] * 23 + [1] * 23, dtype=np.int8)
    for i in range(22):
        side = 0 if i < 11 else 1
        cut = np.sort(rng.choice(np.arange(100, n - 100), 2, replace=False))
        pieces = [(0, cut[0]), (cut[0] + 30, cut[1]), (cut[1] + 10, n)]   # gaps of 1.2 s and 0.4 s
        free = [s for s in range(23 * side, 23 * side + 23)]
        for k, (a, b) in enumerate(pieces):
            s = free[(i % 11) * 2 % 23 if k == 0 else (i % 11 + 11 + k) % 23]
            slots[a:b, s] = true[a:b, i]
    return true, slots, team


@unittest.skipUnless(HAS_SCIPY, "v6 dependency (requirements-v6.txt)")
class TestRetrack(unittest.TestCase):
    def test_continuous_and_close_to_the_players(self):
        true, slots, team = _players()
        tracks, tteam = retrack_all(slots.astype(np.float32), team, [(0, 1500)], 25.0)
        self.assertEqual(tracks.shape, (1500, 22, 2))
        self.assertEqual(list(tteam), [0] * 11 + [1] * 11)
        self.assertFalse(np.isnan(tracks).any())
        step = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
        self.assertLess(step.max(), 3.0)                     # no teleports
        for side in (0, 1):
            d = np.linalg.norm(tracks[:, None, side * 11:side * 11 + 11, :]
                               - true[:, side * 11:side * 11 + 11, None, :], axis=3)
            self.assertLess(np.median(d.min(axis=2)), 0.5)   # every true player is covered

    def test_nan_outside_blocks(self):
        _, slots, team = _players(n=600)
        tracks, _ = retrack_all(slots.astype(np.float32), team, [(50, 550)], 25.0)
        self.assertTrue(np.isnan(tracks[:50]).all() and np.isnan(tracks[550:]).all())
        self.assertFalse(np.isnan(tracks[50:550]).any())


if __name__ == "__main__":
    unittest.main()
