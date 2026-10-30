"""PathCRF evidence: runs on CPU, deterministic, zero outside alive stretches. Skipped without
torch (requirements-v6.txt)."""
import importlib.util
import unittest

import numpy as np

from possession import A, D, H
from possession.tracking import Tracking

HAS_TORCH = all(importlib.util.find_spec(m) is not None
                for m in ("torch", "torch_geometric", "scipy", "matplotlib", "tqdm"))


def _match(seconds=40, seed=0):
    rng = np.random.RandomState(seed)
    n = int(seconds * 25)
    t = np.arange(n) / 25.0
    players = np.zeros((2 * n, 22, 2), dtype=np.float32)
    for i in range(22):
        base = np.array([8 + 4.5 * (i % 11) + (5 if i >= 11 else 0), 6 + 5.5 * (i % 11)])
        path = base + np.stack([4 * np.sin(t / (2 + i % 4) + i), 3 * np.cos(t / (3 + i % 5))], 1)
        players[:n, i] = path
        players[n:, i] = path[::-1]
    ball = players[:, rng.randint(0, 22, 2 * n) % 22].mean(axis=1)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    trk = Tracking(ball=ball.astype(np.float32), players=players, team=team, fps=25.0,
                   pitch=(105.0, 68.0), blocks=[(0, n), (n, 2 * n)])
    labels = np.where((np.arange(2 * n) // 200) % 2 == 0, H, A).astype(np.int8)
    labels[n - 100:n + 100] = D
    return trk, labels


@unittest.skipUnless(HAS_TORCH, "PathCRF needs torch, torch_geometric, scipy, matplotlib, tqdm")
class TestPathCRF(unittest.TestCase):
    def test_deterministic_and_zero_outside_alive(self):
        from possession import pathcrf
        trk, labels = _match()
        cfg = {"weight": 1.0, "trial": 140, "threads": 2, "digits": 3, "clip": 4.0}
        first, info = pathcrf.team_evidence(trk, labels, cfg)
        second, _ = pathcrf.team_evidence(trk, labels, cfg)
        self.assertEqual(first.shape, (trk.n,))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first[labels == D] == 0).all())
        self.assertLessEqual(np.abs(first).max(), 4.0)
        self.assertGreater(info["coverage_of_alive"], 0.5)


if __name__ == "__main__":
    unittest.main()
