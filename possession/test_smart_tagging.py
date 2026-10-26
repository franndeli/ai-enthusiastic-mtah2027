import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession import smart_tagging as st
from possession.frames import A, D, H, X

FPS = 25.0
N = 2000
CAL = st.Calibration(dead_share=0.2, phase_agree=0.8, restart_dead=0.7, restart_agree=0.6)


def inst(code, start, stop, team="Barcelona", **tags):
    return {"code": code, "start_frame": start, "end_frame": stop, "tags": {"Team": team, **tags}}


class EvidenceTest(unittest.TestCase):
    def test_phase_owner_gets_positive_evidence_only_inside_its_half_open_span(self):
        ev = st.log_evidence([inst("BUILD UP", 100, 200)], N, CAL, FPS)
        self.assertEqual(ev.shape, (N, 3))
        np.testing.assert_allclose(ev[100:200], [[np.log(1.6), np.log(0.4), 0.0]] * 100)
        self.assertFalse(ev[:100].any() or ev[200:].any())

    def test_defending_code_names_the_team_without_the_ball(self):
        ev = st.log_evidence([inst("DEFENDING IN MIDDLE THIRD", 0, 50, team="Atletic Club")],
                             N, CAL, FPS)
        self.assertTrue((ev[:50, H] > 0).all() and (ev[:50, A] < 0).all())

    def test_mirrored_twin_is_unioned_not_double_counted(self):
        alone = st.log_evidence([inst("PROGRESSION", 10, 90)], N, CAL, FPS)
        both = st.log_evidence([inst("PROGRESSION", 10, 90),
                                inst("DEFENDING IN MIDDLE THIRD", 10, 90, team="Atletic Club")],
                               N, CAL, FPS)
        np.testing.assert_array_equal(alone, both)

    def test_conflicting_claims_cancel_between_teams(self):
        ev = st.log_evidence([inst("BUILD UP", 0, 10), inst("BUILD UP", 0, 10, team="Atletic Club")],
                             N, CAL, FPS)
        np.testing.assert_allclose(ev[:10, H], ev[:10, A])

    def test_no_evidence_from_na_team_zone_codes_goals_or_unknown_codes(self):
        ev = st.log_evidence([inst("BUILD UP", 0, 100, team="N/A"),
                              inst("BALL IN FINAL THIRD", 0, 100),
                              inst("PLAYERS IN THE BOX", 0, 100),
                              inst("GOALS", 0, 100),
                              inst("SOMETHING NEW", 0, 100)], N, CAL, FPS)
        self.assertFalse(ev.any())

    def test_codes_and_team_names_match_case_insensitively(self):
        ev = st.log_evidence([inst(" build up ", 0, 10, team=" BARCELONA ")], N, CAL, FPS)
        self.assertTrue((ev[:10, H] > 0).all())

    def test_set_piece_dead_before_restart_then_tagged_team(self):
        ev = st.log_evidence([inst("SET PIECES", 1000, 1875, team="Atletic Club",
                                   Type="Corner Kick")], N, CAL, FPS)
        dead = ev[900:1100]  # start - 4 s .. start + 4 s
        self.assertTrue((dead[:, D] > 0).all() and (dead[:, H] < 0).all())
        np.testing.assert_array_equal(dead[:, H], dead[:, A])
        team = ev[1150:1400]  # start + 6 s .. start + 16 s
        np.testing.assert_allclose(team, [[np.log(0.8), np.log(1.2), 0.0]] * 250)
        self.assertFalse(ev[:900].any() or ev[1100:1150].any() or ev[1400:].any())

    def test_throw_in_without_team_gives_dead_ball_evidence_only(self):
        ev = st.log_evidence([inst("SET PIECES", 1000, 1875, team="N/A", Type="Throw-in")],
                             N, CAL, FPS)
        self.assertTrue((ev[900:1100, D] > 0).all())
        self.assertFalse(ev[1100:].any())

    def test_spans_are_clipped_to_the_timeline(self):
        ev = st.log_evidence([inst("SET PIECES", 50, 925, team="N/A"),
                              inst("BUILD UP", N - 5, N + 100)], N, CAL, FPS)
        self.assertTrue((ev[:150, D] > 0).all())
        self.assertTrue((ev[N - 5:, H] > 0).all())

    def test_seconds_in_the_xml_map_to_round_seconds_times_fps(self):
        xml = ("<file><ALL_INSTANCES><instance><ID>1</ID><start>4.01</start><end>6.03</end>"
               "<code>BUILD UP</code><label><group>Team</group><text>Barcelona</text></label>"
               "</instance></ALL_INSTANCES></file>")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "tactical"
            folder.mkdir()
            (folder / "BAR-ATH_tactical_smart_tagging.xml").write_text(xml)
            instances = st.load_instances("BAR-ATH", data_dir=tmp)
        ev = st.log_evidence(instances, N, CAL, FPS)
        np.testing.assert_array_equal(np.flatnonzero(ev[:, H] > 0), np.arange(100, 151))


class CalibrateTest(unittest.TestCase):
    def test_rates_are_shrunk_by_tagged_intervals(self):
        gt = np.full(N, X, np.int8)
        gt[100:500] = D
        gt[500:900] = H
        gt[900:1100] = A
        instances = [inst("PROGRESSION", 500, 1100)]  # 1 interval, 400/600 agree
        cal = st.calibrate(instances, gt, FPS)
        self.assertAlmostEqual(cal.dead_share, 0.4)
        self.assertAlmostEqual(cal.phase_agree, (2 / 3 + 0.5 * 2) / 3)
        self.assertEqual(cal.restart_agree, 0.5)   # no set piece: no information
        self.assertAlmostEqual(cal.restart_dead, 0.4)

    def test_set_piece_rates(self):
        gt = np.full(N, X, np.int8)
        gt[100:500] = D
        gt[500:900] = H
        gt[900:1100] = A
        cal = st.calibrate([inst("SET PIECES", 600, 1475, team="Atletic Club")], gt, FPS)
        self.assertAlmostEqual(cal.restart_dead, (0.0 + 0.4 * 2) / 3)   # 500-700 all alive
        self.assertAlmostEqual(cal.restart_agree, (0.4 + 0.5 * 2) / 3)  # 750-1000: 100/250 A

    def test_needs_alive_and_dead_frames(self):
        gt = np.full(N, H, np.int8)
        with self.assertRaises(ValueError):
            st.calibrate([], gt, FPS)


if __name__ == "__main__":
    unittest.main()
