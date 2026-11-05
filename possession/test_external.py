"""Synthetic-input tests for possession.external: conversion, kloppy loading, noise injection."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession import A, D, H, X
from possession import external
from possession.io import AWAY, HOME, Match

META = """<PutDataRequest><MatchInformation>
<General KickoffTime="2024-01-01T15:30:00+01:00" MatchDay="1" MatchId="MX" Result="1:0"/>
<Environment PitchX="100.0" PitchY="50.0"/>
<OtherGameInformation TotalTimeFirstHalf="2000" TotalTimeSecondHalf="2000"/>
<Teams>
<Team Role="home" TeamId="TH" TeamName="Home"><TrainerStaff/><Players>{home}</Players></Team>
<Team Role="guest" TeamId="TA" TeamName="Away"><TrainerStaff/><Players>{away}</Players></Team>
</Teams></MatchInformation></PutDataRequest>"""
PLAYER = ('<Player PersonId="{pid}" ShirtNumber="1" Shortname="s" FirstName="f" LastName="l" '
          'Starting="true"/>')


def frame(n, x, y, extra=""):
    return '<Frame N="{}" X="{}" Y="{}" S="0" {}/>'.format(n, x, y, extra)


class ConversionTest(unittest.TestCase):
    def test_speed_central_difference(self):
        xy = np.zeros((6, 2, 2), dtype=np.float32)
        xy[:, 0, 0] = np.arange(6) * 0.08  # 0.08 m/frame = 2 m/s at 25 fps
        speed = external.speed_m_s(xy, 25)
        np.testing.assert_allclose(speed[1:-1, 0], 2.0, rtol=1e-5)
        self.assertTrue(np.isnan(speed[0]).all() and np.isnan(speed[-1]).all())
        np.testing.assert_allclose(speed[1:-1, 1], 0.0)

    def test_nan_gap_propagates(self):
        xy = np.zeros((5, 2), dtype=np.float32)
        xy[2] = np.nan
        self.assertTrue(np.isnan(external.speed_m_s(xy, 25)[[1, 2, 3]]).all())  # ends, the gap, and its neighbours

    def test_label_codes(self):
        alive = np.array([True, True, False, False])
        home = np.array([True, False, True, False])
        np.testing.assert_array_equal(external.label_codes(alive, home), [H, A, D, D])

    def test_assemble_places_frames_and_halves(self):
        frame_id = np.array([10, 11, 12, 20, 21])
        period = np.array([1, 1, 1, 2, 2])
        alive = np.array([True, True, False, True, True])
        home = np.array([True, False, True, False, False])
        ball = np.tile(np.array([[0.6, 0.6]]), (5, 1)).astype(np.float32)
        players = np.zeros((5, 2, 2), dtype=np.float32)
        ext = external.assemble(frame_id, period, alive, home, ball, players,
                                np.array([HOME, AWAY]), (100.0, 50.0), 25,
                                np.zeros((0, 3), dtype=np.int64))
        m = ext.match
        self.assertEqual(m.n_frames, 12)
        self.assertEqual(m.halves, [(1, 0, 2), (2, 10, 11)])
        np.testing.assert_array_equal(ext.codes, [H, A, D] + [X] * 7 + [A, A])
        np.testing.assert_allclose(m.ball_xy[0], [0.6, 0.6])
        self.assertTrue(np.isnan(m.ball_xy[5]).all())
        self.assertTrue(np.isnan(m.players_speed[5]).all())


class LoadTest(unittest.TestCase):
    def test_kloppy_files_round_trip(self):
        meta = META.format(home=PLAYER.format(pid="P1"), away=PLAYER.format(pid="P2"))

        def frameset(section, team, pid, frames):
            return '<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">{}</FrameSet>'.format(
                section, team, pid, "".join(frames))

        sets = []
        for section, first in (("firstHalf", 10000), ("secondHalf", 100000)):
            ns = range(first, first + 4)
            status = [1, 1, 0, 0]
            owner = [1, 2, 0, 1]
            sets.append(frameset(section, "BALL", "ball", [
                frame(n, 10 + i, 5, 'Z="0" BallStatus="{}" BallPossession="{}"'.format(s, o))
                for i, (n, s, o) in enumerate(zip(ns, status, owner))]))
            sets.append(frameset(section, "TH", "P1", [frame(n, -20, 0) for n in ns]))
            sets.append(frameset(section, "TA", "P2", [frame(n, 20, 0) for n in ns]))
        tracking = "<MatchTracking>{}</MatchTracking>".format("".join(sets))

        with tempfile.TemporaryDirectory() as tmp:
            meta_path, track_path = Path(tmp, "m_meta.xml"), Path(tmp, "m_tracking.xml")
            meta_path.write_text(meta)
            track_path.write_text(tracking)
            ext = external.load_from_files(meta_path, track_path)
            cache = Path(tmp, "m.npz")
            external.save_cache(cache, ext)
            again = external.load_cache(cache)

        m = ext.match
        self.assertEqual((m.n_frames, m.n_slots, ext.fps), (90004, 2, 25))
        self.assertEqual(m.halves, [(1, 0, 3), (2, 90000, 90003)])
        self.assertEqual(m.pitch_m, (100.0, 50.0))
        np.testing.assert_array_equal(m.team_of_slot, [HOME, AWAY])
        # alive+possession 1 -> H, alive+2 -> A, dead -> D regardless of possession value
        np.testing.assert_array_equal(ext.codes[:4], [H, A, D, D])
        np.testing.assert_allclose(m.players_xy[0], [[0.3, 0.5], [0.7, 0.5]])
        self.assertEqual(ext.raw_ball_counts.tolist(),
                         [[0, 0, 2], [0, 1, 2], [1, 1, 2], [1, 2, 2]])
        np.testing.assert_array_equal(again.codes, ext.codes)
        np.testing.assert_array_equal(again.match.halves, ext.match.halves)


def synthetic_match(n_half=6000, per_team=11, seed=0) -> Match:
    """11 v 11 plus one home substitute who replaces the last home slot at frame 3000."""
    rng = np.random.RandomState(seed)
    n = 2 * n_half + 500
    xy = np.full((n, 2 * per_team + 1, 2), np.nan, dtype=np.float32)
    xy[:, :2 * per_team] = rng.rand(n, 2 * per_team, 2)
    xy[3000:, per_team - 1] = np.nan
    xy[:3000, 2 * per_team] = np.nan
    xy[3000:, 2 * per_team] = rng.rand(n - 3000, 2)
    team = np.array([HOME] * per_team + [AWAY] * per_team + [HOME])
    halves = [(1, 0, n_half - 1), (2, n_half + 500, n - 1)]
    return Match(players_xy=xy, players_speed=np.abs(rng.randn(n, 2 * per_team + 1)).astype(np.float32),
                 ball_xy=rng.rand(n, 2).astype(np.float32),
                 ball_speed=np.abs(rng.randn(n)).astype(np.float32),
                 team_of_slot=team, halves=halves, pitch_m=(105.0, 68.0))


class NoiseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = synthetic_match()
        cls.noisy = external.inject_noise(cls.clean, seed=1)

    def test_dropout_mask_statistics(self):
        rng = np.random.RandomState(3)
        mask = external.dropout_mask(2_000_000, 0.41, 267.5, 850.0, rng)
        self.assertAlmostEqual(float(mask.mean()), 0.41, delta=0.03)
        runs = external.missing_runs_s(mask, [(1, 0, len(mask) - 1)], 25) * 25
        self.assertAlmostEqual(float(np.median(runs)), 267.5, delta=0.1 * 267.5)
        self.assertAlmostEqual(float(np.percentile(runs, 90)), 850.0, delta=0.1 * 850.0)

    def test_ball_dropout_only_inside_halves(self):
        report = external.noise_report(self.clean, self.noisy)
        self.assertAlmostEqual(report["ball_missing_share"], 0.41, delta=0.1)
        self.assertEqual(report["ball_missing_clean_share"], 0.0)
        gap = slice(self.clean.halves[0][2] + 1, self.clean.halves[1][1])
        self.assertFalse(np.isnan(self.noisy.ball_xy[gap, 0]).any())  # only graded frames drop

    def test_ball_speed_nan_where_ball_missing(self):
        missing = np.isnan(self.noisy.ball_xy[:, 0])
        self.assertTrue(np.isnan(self.noisy.ball_speed[missing]).all())

    def test_deterministic_and_seed_sensitive(self):
        again = external.inject_noise(self.clean, seed=1)
        other = external.inject_noise(self.clean, seed=2)
        np.testing.assert_array_equal(again.players_xy, self.noisy.players_xy)
        np.testing.assert_array_equal(again.ball_xy, self.noisy.ball_xy)
        self.assertFalse(np.array_equal(other.ball_xy, self.noisy.ball_xy, equal_nan=True))

    def test_slots_and_teams(self):
        self.assertEqual(self.noisy.n_slots, 46)
        np.testing.assert_array_equal(self.noisy.team_of_slot, [HOME] * 23 + [AWAY] * 23)

    def test_fragmentation_preserves_tracks_without_duplicates(self):
        params = external.NoiseParams(duplicate_prob=0.0, ball_jitter_m=0.0)
        noisy = external.inject_noise(self.clean, seed=4, params=params)
        for team in (HOME, AWAY):
            before = self.clean.players_xy[:, self.clean.team_of_slot == team, 0]
            after = noisy.players_xy[:, noisy.team_of_slot == team, 0]
            np.testing.assert_array_equal(np.sort(np.nan_to_num(before, nan=-1), axis=1)[:, -11:],
                                          np.sort(np.nan_to_num(after, nan=-1), axis=1)[:, -11:])
            np.testing.assert_array_equal((~np.isnan(before)).sum(1), (~np.isnan(after)).sum(1))
        # slot identity is scrambled: some clean track is split over more than one slot
        used = (~np.isnan(noisy.players_xy[..., 0])).any(axis=0).sum()
        self.assertGreater(used, 22)

    def test_duplicates_add_tracks(self):
        params = external.NoiseParams(duplicate_prob=1.0)
        noisy = external.inject_noise(self.clean, seed=5, params=params)
        self.assertGreater(np.sum(~np.isnan(noisy.players_xy[..., 0])),
                           1.9 * np.sum(~np.isnan(self.clean.players_xy[..., 0])))

    def test_too_many_concurrent_tracks_raises(self):
        params = external.NoiseParams(n_slots_per_team=5)
        with self.assertRaises(ValueError):
            external.inject_noise(self.clean, seed=1, params=params)


class TrackerNoiseTest(unittest.TestCase):
    """W7 judge: the target tracker's ball low-pass and wobble on top of the matched noise."""

    @classmethod
    def setUpClass(cls):
        cls.clean = synthetic_match()

    def noisy(self, **extra):
        return external.inject_matched_noise(self.clean, 3, 0.2, 25, 0.3, **extra)

    def test_smoothing_never_bridges_a_gap(self):
        ball = np.column_stack([np.arange(100.0), np.zeros(100)])
        ball[40:60] = np.nan
        out = external.smooth_ball_runs(ball, 4.0)
        np.testing.assert_array_equal(np.isnan(out[:, 0]), np.isnan(ball[:, 0]))
        # the far side of the gap cannot pull the near side: a run's values stay within that run
        self.assertLessEqual(np.nanmax(out[:40, 0]), 39.0)
        self.assertGreaterEqual(np.nanmin(out[60:, 0]), 60.0)

    def test_smoothing_removes_kicks(self):
        speed = np.concatenate([np.full(200, 0.1), np.full(200, -0.5)])     # a sharp change of velocity
        ball = np.column_stack([np.cumsum(speed), np.zeros(400)])
        second = lambda b: np.max(np.abs(b[2:, 0] - 2 * b[1:-1, 0] + b[:-2, 0]))
        self.assertLess(second(external.smooth_ball_runs(ball, 6.0)), 0.2 * second(ball))

    def test_defaults_reproduce_the_matched_noise(self):
        base = self.noisy()
        same = self.noisy(smooth_frames=0.0, wobble_m=0.0)
        np.testing.assert_array_equal(base.ball_xy, same.ball_xy)

    def test_same_dropout_and_tracks_as_the_matched_noise_of_that_seed(self):
        """The new terms draw from their own stream: a seed keeps its gaps and its player tracks."""
        base, tracker = self.noisy(), self.noisy(**external.TRACKER_NOISE)
        np.testing.assert_array_equal(np.isnan(base.ball_xy[:, 0]), np.isnan(tracker.ball_xy[:, 0]))
        np.testing.assert_array_equal(base.players_xy, tracker.players_xy)
        np.testing.assert_array_equal(base.players_speed, tracker.players_speed)
        self.assertFalse(np.array_equal(base.ball_xy, tracker.ball_xy, equal_nan=True))

    def test_deterministic_and_ball_speed_follows_the_new_ball(self):
        a, b = self.noisy(**external.TRACKER_NOISE), self.noisy(**external.TRACKER_NOISE)
        np.testing.assert_array_equal(a.ball_xy, b.ball_xy)
        np.testing.assert_array_equal(np.isnan(a.ball_speed), np.isnan(a.ball_speed) | np.isnan(a.ball_xy[:, 0]))

    def test_provider_helper_uses_its_noise_settings(self):
        ext = external.ExternalMatch(self.clean, np.zeros(self.clean.n_frames, dtype=np.int8), 25, np.zeros((0, 3)))
        noise = external.MATCHED_NOISE["idsse"]
        direct = external.inject_matched_noise(self.clean, 1, noise["share"], 25, noise["ball_err_m"], **external.TRACKER_NOISE)
        np.testing.assert_array_equal(external.inject_tracker_noise(ext, "idsse", 1).ball_xy, direct.ball_xy)


if __name__ == "__main__":
    unittest.main()
