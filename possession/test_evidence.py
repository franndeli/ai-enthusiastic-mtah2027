"""Synthetic checks of the evidence rules; never reads match assets."""
import unittest

import numpy as np

from possession import A, D, H
from possession import evidence as E
from possession import features as F
from possession.io import AWAY, HOME, Match


def flat_features(n):
    """Features of an unremarkable stretch: nobody controls, players at the neutral speed, ball inside."""
    return {
        "ball_present": np.ones(n, np.float32),
        "control_balance": np.zeros(n, np.float32),
        "last_team_sign": np.zeros(n, np.float32),
        "next_team_sign": np.zeros(n, np.float32),
        "frames_since_team": np.full(n, np.inf, np.float32),
        "frames_until_team": np.full(n, np.inf, np.float32),
        "all_mean_speed_smooth_m_s": np.full(n, 1.5, np.float32),
        "ball_out_frac": np.zeros(n, np.float32),
    }


def team_pref(ev):
    """H minus A log-evidence per frame (>0 favours home)."""
    return ev[:, H] - ev[:, A]


class TeamEvidenceTests(unittest.TestCase):
    n = 50

    def test_control_gives_team_evidence(self):
        f = flat_features(self.n)
        f["control_balance"][:] = 0.9
        self.assertTrue((team_pref(E.scores(f)) > 0).all())
        f["control_balance"][:] = -0.9
        self.assertTrue((team_pref(E.scores(f)) < 0).all())

    def test_no_clue_is_flat_and_alive(self):
        ev = E.scores(flat_features(self.n))
        np.testing.assert_allclose(team_pref(ev), 0.0, atol=1e-6)
        self.assertTrue((ev[:, D] < ev[:, H]).all())

    def test_missing_ball_with_no_memory_is_flat(self):
        f = flat_features(self.n)
        f["ball_present"][:] = 0
        f["control_balance"][:] = np.nan
        ev = E.scores(f)
        self.assertFalse(np.isnan(ev).any())
        np.testing.assert_allclose(team_pref(ev), 0.0, atol=1e-6)

    def test_gap_memory_places_handover_in_the_middle(self):
        n = 101
        f = flat_features(n)
        f["last_team_sign"][:] = 1.0     # home had it before the gap
        f["next_team_sign"][:] = -1.0    # away has it after
        f["frames_since_team"] = np.arange(n, dtype=np.float32)
        f["frames_until_team"] = np.arange(n - 1, -1, -1, dtype=np.float32)
        pref = team_pref(E.scores(f))
        self.assertGreater(pref[5], 0)
        self.assertLess(pref[95], 0)
        self.assertAlmostEqual(float(pref[50]), 0.0, places=5)

    def test_gap_memory_is_off_where_control_is_clear(self):
        f = flat_features(self.n)
        f["control_balance"][:] = -0.9   # clear away control now
        f["last_team_sign"][:] = 1.0
        f["frames_since_team"][:] = 1.0
        self.assertTrue((team_pref(E.scores(f)) < 0).all())

    def test_team_confidence_is_capped(self):
        f = flat_features(self.n)
        f["control_balance"][:] = 1.0
        cap = np.log(0.75 / 0.25)
        pref = team_pref(E.scores(f, {"w_control": 50.0}))
        np.testing.assert_allclose(pref, cap, rtol=1e-4)

    def test_team_confidence_never_decides_dead(self):
        f = flat_features(self.n)
        d_flat = E.scores(f)[:, D]
        f["control_balance"][:] = 0.9
        d_ctrl = E.scores(f)[:, D]
        np.testing.assert_allclose(d_flat, d_ctrl)
        ev = E.scores(f)
        np.testing.assert_allclose(ev[:, H], np.log(1 - np.exp(d_ctrl)), rtol=1e-4)


class ShapeEvidenceTests(unittest.TestCase):
    n = 50

    def shape_features(self, **z):
        f = flat_features(self.n)
        for name in ("len", "width", "speed", "vel"):
            f["shape_%s_z" % name] = np.full(self.n, z.get(name, 0.0), np.float32)
        return f

    def test_longer_wider_home_team_is_home_evidence(self):
        pref = team_pref(E.scores(self.shape_features(len=2.0, width=1.0)))
        self.assertTrue((pref > 0).all())
        pref = team_pref(E.scores(self.shape_features(len=-2.0, width=-1.0)))
        self.assertTrue((pref < 0).all())

    def test_faster_home_team_is_away_evidence(self):
        self.assertTrue((team_pref(E.scores(self.shape_features(speed=2.0))) < 0).all())

    def test_shape_speaks_while_ball_is_unseen(self):
        f = self.shape_features(len=2.0)
        f["ball_present"][:] = 0
        f["control_balance"][:] = np.nan
        self.assertTrue((team_pref(E.scores(f)) > 0).all())

    def test_shape_is_switchable_and_absent_features_are_ignored(self):
        f = self.shape_features(len=3.0, vel=3.0)
        np.testing.assert_allclose(team_pref(E.scores(f, {"w_shape": 0.0})), 0.0, atol=1e-6)
        self.assertEqual(E.scores(flat_features(3)).shape, (3, 3))

    def test_shape_is_capped_like_any_team_evidence(self):
        pref = team_pref(E.scores(self.shape_features(len=3.0, width=3.0), {"w_shape": 50.0}))
        np.testing.assert_allclose(pref, np.log(0.75 / 0.25), rtol=1e-4)

    def test_restart_team_evidence_is_switchable(self):
        f = flat_features(self.n)
        f["restart_team"] = np.full(self.n, -1.0, np.float32)
        np.testing.assert_allclose(team_pref(E.scores(f)), 0.0, atol=1e-6)
        self.assertTrue((team_pref(E.scores(f, {"w_restart_team": 1.0})) < 0).all())


class DeadEvidenceTests(unittest.TestCase):
    n = 50

    def test_players_walking_is_dead(self):
        f = flat_features(self.n)
        f["all_mean_speed_smooth_m_s"][:] = 0.8
        ev = E.scores(f)
        self.assertTrue((ev[:, D] > ev[:, H]).all())

    def test_running_play_is_alive(self):
        f = flat_features(self.n)
        f["all_mean_speed_smooth_m_s"][:] = 3.5
        ev = E.scores(f)
        self.assertTrue((ev[:, D] < ev[:, H]).all())

    def test_ball_out_raises_dead(self):
        f = flat_features(self.n)
        base = E.scores(f)[:, D]
        f["ball_out_frac"][:] = 1.0
        self.assertTrue((E.scores(f)[:, D] > base).all())

    def test_tag_is_capped_prior_not_label(self):
        f = flat_features(self.n)
        f["tag_set_piece"] = np.ones(self.n, np.float32)
        ev = E.scores(f)
        self.assertTrue((ev[:, D] > E.scores(flat_features(self.n))[:, D]).all())
        # alone, a tag does not flip a running, ball-in-play frame to dead
        self.assertTrue((ev[:, H] > ev[:, D]).all())
        # even a huge weight cannot exceed the dead probability cap
        huge = E.scores(f, {"w_tag_dead": 100.0})
        self.assertLessEqual(float(np.exp(huge[:, D]).max()), 0.8 + 1e-6)

    def test_missing_tags_key_is_fine(self):
        self.assertNotIn("tag_set_piece", flat_features(3))
        self.assertEqual(E.scores(flat_features(3)).shape, (3, 3))

    def test_nan_features_never_give_nan(self):
        f = flat_features(self.n)
        for key in ("control_balance", "all_mean_speed_smooth_m_s", "ball_out_frac"):
            f[key][:] = np.nan
        ev = E.scores(f)
        self.assertFalse(np.isnan(ev).any())
        self.assertEqual(ev.dtype, np.float32)

    def test_weight_zero_switches_clue_off(self):
        f = flat_features(self.n)
        f["all_mean_speed_smooth_m_s"][:] = 0.5
        off = E.scores(f, {"w_slow": 0.0})
        np.testing.assert_allclose(off, E.scores(flat_features(self.n), {"w_slow": 0.0}))
        self.assertFalse(np.allclose(off, E.scores(f)))


class DeadOnsetEvidenceTests(unittest.TestCase):
    n = 50

    def test_lead_speed_brings_dead_evidence_forward(self):
        f = flat_features(self.n)
        f["all_mean_speed_smooth_m_s"][:] = 3.5      # still running now
        f["all_mean_speed_lead_m_s"] = np.full(self.n, 0.8, np.float32)  # but stopping within 3 s
        self.assertTrue((E.scores(f)[:, D] > E.scores(f)[:, H]).all())
        self.assertTrue((E.scores(f, {"slow_use_lead": 0})[:, D] < E.scores(f)[:, H]).all())

    def test_exited_ball_raises_dead_and_is_switchable(self):
        f = flat_features(self.n)
        base = E.scores(f)[:, D]
        f["ball_exited"] = np.ones(self.n, np.float32)
        self.assertTrue((E.scores(f)[:, D] > base).all())
        np.testing.assert_allclose(E.scores(f, {"w_exit": 0.0})[:, D], base)

    def test_long_gap_with_slow_players_is_dead_evidence_only_when_asked(self):
        f = flat_features(self.n)
        f["all_mean_speed_lead_m_s"] = np.full(self.n, 0.8, np.float32)
        f["ball_present"][:] = 0
        f["frames_since_ball"] = np.full(self.n, 200, np.float32)
        f["frames_until_ball"] = np.full(self.n, 200, np.float32)
        off = E.scores(f)[:, D]
        on = E.scores(f, {"w_gap_dead": 1.0})[:, D]
        self.assertTrue((on > off).all())
        f["all_mean_speed_lead_m_s"][:] = 3.5          # players running: gap says nothing
        np.testing.assert_allclose(E.scores(f, {"w_gap_dead": 1.0})[:, D], E.scores(f)[:, D])


class EndToEndTests(unittest.TestCase):
    def test_home_carries_then_away_takes_over_then_stoppage(self):
        """Ball glued to the home player, then to the away player, then everyone walks."""
        n = 600
        players = np.zeros((n, 2, 2), np.float32)
        players[:, 0] = (0.4, 0.5)
        players[:, 1] = (0.8, 0.5)
        ball = np.zeros((n, 2), np.float32)
        ball[:200] = (0.402, 0.5)
        ball[200:400] = (0.802, 0.5)
        ball[400:] = (0.6, 0.5)
        speed = np.full((n, 2), 3.0, np.float32)
        speed[400:] = 0.6
        match = Match(players_xy=players, players_speed=speed, ball_xy=ball,
                      ball_speed=np.full(n, np.nan, np.float32),
                      team_of_slot=np.array([HOME, AWAY], np.int8), halves=[(1, 0, n - 1)],
                      pitch_m=(100.0, 50.0))
        ev = E.scores(F.compute(match, tags=[]))
        label = np.argmax(ev, axis=1)
        self.assertTrue((label[40:160] == H).all())
        self.assertTrue((label[240:300] == A).all())
        self.assertTrue((label[440:560] == D).all())


if __name__ == "__main__":
    unittest.main()
