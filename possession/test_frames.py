import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession.frames import AWAY, HOME, FrameTable, period_runs


def table(n=6, k=2, labels=True):
    return FrameTable(fps=25.0, pitch_m=(105.0, 68.0),
                      period=np.array([1] * (n // 2) + [2] * (n - n // 2), np.int8),
                      ball_xy_m=np.zeros((n, 2), np.float32),
                      players_xy_m=np.zeros((n, k, 2), np.float32),
                      slot_team=np.array([HOME, AWAY][:k], np.int8),
                      labels=np.zeros(n, np.int8) if labels else None)


class FrameTableTest(unittest.TestCase):
    def test_round_trip(self):
        original = table()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.npz"
            original.save(path)
            loaded = FrameTable.load(path)
        self.assertEqual(loaded.pitch_m, (105.0, 68.0))
        np.testing.assert_array_equal(loaded.labels, original.labels)
        np.testing.assert_array_equal(loaded.period, original.period)

    def test_round_trip_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.npz"
            table(labels=False).save(path)
            self.assertIsNone(FrameTable.load(path).labels)

    def test_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            FrameTable(25.0, (105, 68), np.ones(3, np.int8), np.zeros((2, 2)),
                       np.zeros((3, 1, 2)), np.zeros(1, np.int8))
        with self.assertRaises(ValueError):
            FrameTable(25.0, (105, 68), np.ones(3, np.int8), np.zeros((3, 2)),
                       np.zeros((3, 1, 2)), np.array([2], np.int8))

    def test_period_runs_split_halves_and_skip_outside_play(self):
        period = np.array([0, 1, 1, 0, 2, 2, 2, 0], np.int8)
        self.assertEqual(period_runs(period), [(1, 3), (4, 7)])
        self.assertEqual(period_runs(np.array([1, 1, 2], np.int8)), [(0, 2), (2, 3)])


if __name__ == "__main__":
    unittest.main()
