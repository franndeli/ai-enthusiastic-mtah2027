"""Synthetic tests of the SkillCorner conversion (no files, no real data)."""
import unittest

import numpy as np

from possession import A, D, H, X
from possession import skillcorner as sc


def _events(rows):
    """rows: (start, end, team, before, after, foul_end[, taker]); taker defaults to the team."""
    cols = list(zip(*[r[:6] for r in rows]))
    taker = [r[6] if len(r) > 6 else r[2] for r in rows]
    return (np.array(cols[0]), np.array(cols[1]), np.array(cols[2], dtype=np.int8),
            np.array(cols[3], dtype=bool), np.array(cols[4], dtype=bool), np.array(cols[5], dtype=bool),
            np.array(taker, dtype=np.int8))


class ResampleTest(unittest.TestCase):
    def test_exact_samples_and_interpolation(self):
        frames = np.array([0, 1, 2])
        values = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
        ticks = np.arange(0, 11, 2)          # 25 fps ticks covering source frames 0..2
        out = sc.resample_positions(frames, values, ticks)[:, 0]
        np.testing.assert_allclose(out, [0, 4, 8, 12, 16, 20], atol=1e-5)

    def test_exact_sample_survives_a_missing_neighbour(self):
        frames = np.array([0, 1, 2])
        values = np.array([[1.0], [np.nan], [3.0]], dtype=np.float32)
        out = sc.resample_positions(frames, values, np.arange(0, 11, 2))[:, 0]
        self.assertEqual(out[0], 1.0)        # exact sample kept although its neighbour is NaN
        self.assertEqual(out[5], 3.0)
        self.assertTrue(np.isnan(out[1:5]).all())

    def test_no_interpolation_across_a_skipped_source_frame(self):
        frames = np.array([0, 2])
        values = np.array([[0.0], [20.0]], dtype=np.float32)
        out = sc.resample_positions(frames, values, np.arange(0, 11, 2))[:, 0]
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[5], 20.0)
        self.assertTrue(np.isnan(out[1:5]).all())

    def test_central_speed(self):
        xy = np.stack([np.arange(5) * 0.4, np.zeros(5)], axis=1)
        speed = sc.central_speed(xy, fps=25)
        np.testing.assert_allclose(speed[1:-1], 10.0, atol=1e-5)
        self.assertTrue(np.isnan(speed[[0, -1]]).all())


class EventLabelTest(unittest.TestCase):
    PITCH = (100.0, 60.0)

    def ball(self, n):
        return np.full((n, 2), np.nan)

    def test_team_is_carried_and_dead_before_first_possession(self):
        ev = _events([(10, 30, H, False, False, False), (50, 60, A, False, False, False)])
        lab = sc.event_labels(100, [(0, 100)], *ev, self.ball(100), self.PITCH)
        self.assertTrue((lab[:10] == D).all())
        self.assertTrue((lab[10:50] == H).all())     # carried through the loose ball 30..49
        self.assertTrue((lab[50:] == A).all())

    def test_stoppage_located_on_the_seen_ball(self):
        ball = self.ball(300)
        ball[40:66] = (50.0, 30.0) + np.arange(26)[:, None] * (0.0, 1.2)   # leaves at y > 60.3
        ball[66:70] = (50.0, 62.0)
        ball[185:196] = (50.0, 60.0)                 # still on the line before the throw
        ball[196:200] = (50.0, 60.0) + np.arange(1, 5)[:, None] * (0.0, -2.0)
        ev = _events([(10, 60, H, False, True, False), (200, 220, A, True, False, False)])
        lab = sc.event_labels(300, [(0, 300)], *ev, ball, self.PITCH)
        first_out = 40 + int(np.argmax(ball[40:70, 1] > 60.3))
        self.assertTrue((lab[60:first_out] == H).all())
        self.assertTrue((lab[first_out:196] == D).all())
        self.assertTrue((lab[196:] == A).all())      # the throw in flight belongs to its taker

    def test_defaults_without_a_seen_ball(self):
        ev = _events([(10, 60, H, False, True, False), (200, 220, A, True, False, False),
                      (230, 240, A, False, True, True), (280, 290, H, True, False, False)])
        lab = sc.event_labels(300, [(0, 300)], *ev, self.ball(300), self.PITCH)
        self.assertTrue((lab[60:70] == H).all() and (lab[70:190] == D).all())
        self.assertTrue((lab[190:230] == A).all())
        self.assertTrue((lab[240:270] == D).all())   # a foul stops play at once
        self.assertTrue((lab[270:] == H).all())

    def test_intercepted_restart_belongs_to_its_taker_until_the_interception(self):
        ev = _events([(10, 60, H, False, True, False), (200, 220, A, True, False, False, H)])
        lab = sc.event_labels(300, [(0, 300)], *ev, self.ball(300), self.PITCH)
        self.assertTrue((lab[70:190] == D).all())
        self.assertTrue((lab[190:200] == H).all() and (lab[200:] == A).all())

    def test_restart_is_own(self):
        self.assertTrue(sc.restart_is_own("throw_in_for"))
        self.assertFalse(sc.restart_is_own("corner_against"))
        self.assertTrue(sc.restart_is_own("goal_against"))   # the conceding team kicks off
        self.assertFalse(sc.restart_is_own("goal_for"))
        self.assertTrue(sc.restart_is_own(""))

    def test_last_interruption_runs_to_the_period_end_and_periods_are_separate(self):
        ev = _events([(10, 60, H, False, True, False), (120, 130, A, False, False, False)])
        lab = sc.event_labels(200, [(0, 100), (110, 200)], *ev, self.ball(200), self.PITCH)
        self.assertTrue((lab[70:100] == D).all())
        self.assertTrue((lab[100:110] == X).all())
        self.assertTrue((lab[110:120] == D).all())   # second period: dead until its first possession
        self.assertTrue((lab[120:] == A).all())

    def test_swapping_teams_swaps_labels(self):
        rows = [(10, 60, H, False, True, False), (150, 170, A, True, False, False)]
        swapped = [(s, e, A if t == H else H, b, a, f) for s, e, t, b, a, f in rows]
        one = sc.event_labels(200, [(0, 200)], *_events(rows), self.ball(200), self.PITCH)
        two = sc.event_labels(200, [(0, 200)], *_events(swapped), self.ball(200), self.PITCH)
        np.testing.assert_array_equal(np.where(one == H, A, np.where(one == A, H, one)), two)


class PhaseLabelTest(unittest.TestCase):
    def test_phases_and_complement(self):
        lab = sc.phase_labels(50, [(5, 45)], [(10, 20, H), (30, 48, A)])
        self.assertTrue((lab[:5] == X).all() and (lab[45:] == X).all())
        self.assertTrue((lab[5:10] == D).all() and (lab[10:20] == H).all())
        self.assertTrue((lab[20:30] == D).all() and (lab[30:45] == A).all())


if __name__ == "__main__":
    unittest.main()
