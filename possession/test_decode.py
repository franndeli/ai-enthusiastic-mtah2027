import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import H, A, D, X  # noqa: E402
from possession.decode import segment, viterbi2  # noqa: E402


def brute(score, c01, c10):
    best, arg = -np.inf, None
    for states in itertools.product((0, 1), repeat=len(score)):
        v = sum(s / 2.0 if q else -s / 2.0 for s, q in zip(score, states))
        for p, q in zip(states, states[1:]):
            if p == 0 and q == 1:
                v -= c01
            elif p == 1 and q == 0:
                v -= c10
        if v > best + 1e-12:
            best, arg = v, states
    return np.array(arg), best


def value(score, states, c01, c10):
    v = sum(s / 2.0 if q else -s / 2.0 for s, q in zip(score, states))
    for p, q in zip(states, states[1:]):
        v -= c01 if (p == 0 and q == 1) else (c10 if (p == 1 and q == 0) else 0.0)
    return v


class TestViterbi(unittest.TestCase):
    def test_matches_exhaustive_search(self):
        rng = np.random.RandomState(0)
        for _ in range(300):
            n = rng.randint(1, 10)
            score = rng.normal(0, 3, n)
            c01, c10 = rng.uniform(0, 6), rng.uniform(0, 6)
            got = viterbi2(score, c01, c10)
            _, best = brute(score, c01, c10)
            self.assertAlmostEqual(value(score, got, c01, c10), best, places=9)

    def test_boundary_at_evidence_change_no_lag(self):
        score = np.array([-1.0] * 50 + [1.0] * 50)
        states = viterbi2(score, 5.0)
        self.assertEqual(int(np.argmax(states)), 50)

    def test_short_blip_absorbed_long_run_kept(self):
        score = np.array([-1.0] * 40 + [1.0] * 3 + [-1.0] * 40 + [1.0] * 30 + [-1.0] * 40)
        states = viterbi2(score, 4.0)
        self.assertTrue((states[:83] == 0).all())
        self.assertTrue((states[83:113] == 1).all())

    def test_segment_two_stages(self):
        n = 400
        dead = np.full(n, -2.0)
        dead[150:200] = 2.0
        team = np.full(n, 0.0)
        team[:150] = 1.0
        team[200:] = -1.0
        lab = segment(dead, team, [(50, 350)], n, cost_dead=5, cost_team=5)
        self.assertTrue((lab[:50] == X).all() and (lab[350:] == X).all())
        self.assertTrue((lab[50:150] == H).all())
        self.assertTrue((lab[150:200] == D).all())
        self.assertTrue((lab[200:350] == A).all())


if __name__ == "__main__":
    unittest.main()
