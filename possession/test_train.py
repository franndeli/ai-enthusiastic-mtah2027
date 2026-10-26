import functools
import unittest
from unittest import mock

import numpy as np

from possession.frames import AWAY, HOME, A, D, H, X, FrameTable
from possession.train import (baseline_codes, drop_ball, mirror, nearest_team_codes, prepare,
                              train)
from possession.model import decode


def match(n=3000, seed=0):
    """Synthetic match: possession alternates every 200 frames, the owner stands on the ball."""
    rng = np.random.RandomState(seed)
    period = np.array([1] * (n // 2) + [0] + [2] * (n - n // 2 - 1), np.int8)
    owner = (np.arange(n) // 200) % 3                    # 0 home, 1 away, 2 dead
    labels = np.where(period == 0, X, owner).astype(np.int8)
    ball = np.stack([np.cumsum(rng.normal(0, 0.1, n)), np.zeros(n)], axis=1)
    home = ball + np.where(owner[:, None] == 0, 0.5, 6.0)
    away = ball - np.where(owner[:, None] == 1, 0.5, 6.0)
    dead = owner == 2
    ball[dead, 1] = 40.0  # out of play while dead
    players = np.stack([home, away], axis=1).astype(np.float32)
    return FrameTable(25.0, (105.0, 68.0), period, ball.astype(np.float32), players,
                      np.array([HOME, AWAY], np.int8), labels)


class TrainHelpersTest(unittest.TestCase):
    def test_drop_ball_rate_blocks_and_determinism(self):
        table = match(n=200000)
        a = drop_ball(table, np.random.RandomState(1), gap_median_s=1.0, gap_sigma=0.5)
        b = drop_ball(table, np.random.RandomState(1), gap_median_s=1.0, gap_sigma=0.5)
        hidden = np.isnan(a.ball_xy_m[:, 0])
        self.assertAlmostEqual(hidden.mean(), 0.407, delta=0.06)
        self.assertLess(np.abs(np.diff(hidden.astype(int))).sum(), 0.1 * len(hidden))  # blocky
        dead = table.labels == D
        self.assertGreater(hidden[dead].mean(), 2 * hidden[~dead & (table.labels != X)].mean())
        np.testing.assert_array_equal(hidden, np.isnan(b.ball_xy_m[:, 0]))
        self.assertFalse(np.isnan(table.ball_xy_m).any())  # original untouched

    def test_mirror_swaps_teams_only(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        flipped, swapped = mirror(values, np.array([H, A, D], np.int8), np.array([True, False]))
        np.testing.assert_array_equal(flipped, [[-1, 2], [-3, 4], [-5, 6]])
        np.testing.assert_array_equal(swapped, [A, H, D])

    def test_nearest_team_codes(self):
        codes = nearest_team_codes(match())
        self.assertEqual(codes[100], HOME)
        self.assertEqual(codes[300], AWAY)
        table = match()
        ball = table.ball_xy_m.copy()
        ball[5] = np.nan
        dropped = FrameTable(25.0, table.pitch_m, table.period, ball, table.players_xy_m,
                             table.slot_team, table.labels)
        self.assertEqual(nearest_team_codes(dropped)[5], -1)
        self.assertEqual(len(baseline_codes(dropped)), table.n_frames)

    def test_pipeline_recovers_synthetic_possessions(self):
        rng = np.random.RandomState(0)
        # 8 s synthetic possessions cannot survive ATD-length gaps; this tests the wiring.
        short_gaps = functools.partial(drop_ball, gap_median_s=1.0, gap_sigma=0.5)
        with mock.patch("possession.train.drop_ball", short_gaps):
            prepared = [prepare(match(seed=s), rng) for s in range(3)]
        model = train(prepared[:2], emission_scale=0.2)
        held = prepared[2]
        pred = decode(model, held["values"], held["table"].period)
        graded = held["table"].labels != X
        accuracy = (pred[graded] == match(seed=2).labels[graded]).mean()
        self.assertGreater(accuracy, 0.85)


if __name__ == "__main__":
    unittest.main()
