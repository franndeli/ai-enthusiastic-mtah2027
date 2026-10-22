"""Synthetic clock recovery and replay rejection; no match assets."""
import unittest
import numpy as np
from possession.io import Match
from possession.broadcast import align, recovered_ball, formation_descriptor


def synthetic(xy):
    n, slots, _ = xy.shape
    return Match(xy, np.zeros((n, slots)), np.full((n, 2), .5), np.zeros(n),
                 np.repeat([0, 1], slots//2), [(1, 0, n-1)], (105, 68))


class BroadcastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(41)
        # Smooth unique motion, with distinct formations every second.
        anchors = rng.uniform(.1, .9, (31, 16, 2))
        xy = np.stack([np.interp(np.arange(600), np.arange(31)*20, anchors[:, s, c])
                       for s in range(16) for c in range(2)], axis=1).reshape(600, 16, 2)
        self.tactical = synthetic(xy)
        self.broadcast = synthetic(xy[140:340, ::-1].copy())
        self.broadcast.team_of_slot = self.tactical.team_of_slot[::-1]

    def test_shift_and_slot_permutation(self):
        result = align(self.tactical, self.broadcast, np.ones(200, bool), stride=10)
        self.assertEqual(result.aligned_segments, 1)
        np.testing.assert_array_equal(result.tactical_frame, np.arange(140, 340))
        ball = recovered_ball(self.tactical, self.broadcast, np.ones(200, bool), stride=10)
        self.assertTrue(np.isfinite(ball[140:340]).all())
        self.assertTrue(np.isnan(ball[:140]).all())

    def test_reversed_replay_rejected(self):
        self.broadcast.players_xy = self.broadcast.players_xy[::-1].copy()
        result = align(self.tactical, self.broadcast, np.ones(200, bool), stride=10)
        self.assertEqual(result.aligned_segments, 0)
        self.assertTrue(np.all(result.tactical_frame == -1))

    def test_team_awareness(self):
        original = formation_descriptor(self.broadcast)
        self.broadcast.team_of_slot = 1-self.broadcast.team_of_slot
        self.assertFalse(np.array_equal(original, formation_descriptor(self.broadcast)))

    def test_invalid_homography_never_recovers(self):
        ball = recovered_ball(self.tactical, self.broadcast, np.zeros(200, bool), stride=10)
        self.assertTrue(np.isnan(ball).all())

    def test_outside_pitch_ball_not_recovered(self):
        self.broadcast.ball_xy[:] = [1.5, .5]
        ball = recovered_ball(self.tactical, self.broadcast, np.ones(200, bool), stride=10)
        self.assertTrue(np.isnan(ball).all())
