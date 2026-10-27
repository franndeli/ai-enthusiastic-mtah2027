import os
import tempfile
import unittest

import numpy as np

from possession import A, D, H, X, bundle
from possession.tracking import Tracking, read_epts


def _match(n1=300, n2=250, pitch=(100.0, 64.0), seed=0):
    rng = np.random.RandomState(seed)
    n = n1 + n2
    players = rng.uniform(5, 60, (n, 30, 2)).astype(np.float32)
    players[:, 28:] = np.nan                       # two squad players who never play
    ball = rng.uniform(5, 60, (n, 2)).astype(np.float32)
    ball[::3] = np.nan
    team = np.array([0] * 15 + [1] * 15, dtype=np.int8)
    labels = rng.choice([H, A, D], n).astype(np.int8)
    trk = Tracking(ball=ball, players=players, team=team, fps=25.0, pitch=pitch,
                   blocks=[(0, n1), (n1, n)])
    return trk, labels


LAY = {"n": 1200, "fps": 25.0, "blocks": [(100, 500), (600, 1000)], "starts": [100, 700],
       "pitch": (105.3, 68.0)}


class TestBundle(unittest.TestCase):
    def test_layout_matches_the_challenge(self):
        lay = bundle.layout()
        self.assertEqual(lay["n"], 199319)
        self.assertEqual(lay["blocks"], [(11185, 90652), (110808, 190198)])
        self.assertEqual(lay["starts"][1] - lay["blocks"][1][0], 2934)

    def test_place(self):
        trk, lab = _match()
        arr = bundle.place(trk, lab, LAY)
        gt = arr["gt"]
        self.assertTrue((gt[:100] == X).all() and (gt[500:600] == X).all() and (gt[1000:] == X).all())
        np.testing.assert_array_equal(gt[100:400], lab[:300])          # half 1 shorter: X tail
        self.assertTrue((gt[400:500] == X).all())
        self.assertTrue((gt[600:700] == D).all())                     # before the 2nd kick-off
        np.testing.assert_array_equal(gt[700:950], lab[300:550])
        self.assertTrue((gt[950:1000] == X).all())
        self.assertEqual(int((arr["slot_team"] == 0).sum()), 23)
        # home source slots 0..14 -> bundle slots 0..14; away 15..27 -> 23..35 (28, 29 never seen)
        np.testing.assert_array_equal(arr["players"][100:400, :15], trk.players[:300, :15])
        np.testing.assert_array_equal(arr["players"][100:400, 23:36], trk.players[:300, 15:28])
        self.assertTrue(np.isnan(arr["players"][:, 36:]).all())

    def test_long_period_is_cut_at_the_half_end(self):
        trk, lab = _match(n1=450, n2=350)
        gt = bundle.place(trk, lab, LAY)["gt"]
        np.testing.assert_array_equal(gt[100:500], lab[:400])
        np.testing.assert_array_equal(gt[700:1000], lab[450:750])

    def test_write_round_trip(self):
        trk, lab = _match()
        arr = bundle.place(trk, lab, LAY)
        with tempfile.TemporaryDirectory() as tmp:
            paths = bundle.write(tmp, arr, LAY, "T")
            back = read_epts(paths["atd_meta"], paths["atd_raw"], pitch=trk.pitch, n_frames=LAY["n"])
            ok = ~np.isnan(arr["players"])
            np.testing.assert_allclose(back.players[ok], arr["players"][ok], atol=1e-3)
            self.assertTrue(np.isnan(back.players[~ok]).all())
            okb = ~np.isnan(arr["ball"])
            np.testing.assert_allclose(back.ball[okb], arr["ball"][okb], atol=1e-3)
            np.testing.assert_array_equal(back.team, arr["slot_team"])
            with open(paths["gt"]) as handle:
                self.assertEqual(len(handle.readlines()), LAY["n"] + 1)
            self.assertTrue(os.path.isfile(paths["smart_tagging"]))

    def test_speeds(self):
        p = np.zeros((50, 1, 2), dtype=np.float32)
        p[:, 0, 0] = np.arange(50) * 0.2                              # 5 m/s at 25 fps
        s = bundle.speeds(p, 25.0, [(0, 50)])
        np.testing.assert_allclose(s[3:47, 0], 5.0, rtol=1e-5)
        self.assertTrue(np.isnan(s[:3]).all() and np.isnan(s[47:]).all())


if __name__ == "__main__":
    unittest.main()
