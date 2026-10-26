import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession.atd import load_atd
from possession.frames import AWAY, HOME

META = """<?xml version="1.0" encoding="utf-8"?>
<Root><Metadata>
  <Teams>
    <Team id="T_HOME"><Name>Home FC</Name></Team>
    <Team id="T_AWAY"><Name>Away FC</Name></Team>
  </Teams>
  <Players>
    <Player id="P1" teamId="T_HOME"><Name>Track 1</Name></Player>
    <Player id="P2" teamId="T_AWAY"><Name>Track 2</Name></Player>
    <Player id="P3" teamId="T_HOME"><Name>Track 3</Name></Player>
  </Players>
</Metadata></Root>
"""

# frame_id 1..6 except 4 (absent). Frame 2: NaN ball; frame 3: slot 2 missing; frame 5: bare NaN group.
RAW = """1:0.5,0.5,0.0;0.0,0.0,1.0;1.0,1.0,2.0:0.25,0.75,3.0
2:0.1,0.2,0.0;0.3,0.4,0.0;0.5,0.6,0.0:NaN,NaN,NaN
3:0.1,0.2,0.0;NaN,NaN,NaN;0.5,0.6,0.0:0.5,0.5,0.0
5:NaN;0.3,0.4,0.0;0.5,0.6,0.0:0.5,0.5,0.0
6:0.5,0.5,0.0;0.5,0.5,0.0;0.5,0.5,0.0:0.5,0.5,0.0
"""

PITCH = (100.0, 60.0)
N = 8
BLOCKS = [(1, 1, 2), (2, 4, 5)]  # rows 0, 3, 6, 7 are outside play


class LoadAtdTest(unittest.TestCase):
    def load(self, raw=RAW, n=N):
        with tempfile.TemporaryDirectory() as tmp:
            meta, raw_path = Path(tmp, "m.xml"), Path(tmp, "r.txt")
            meta.write_text(META)
            raw_path.write_text(raw)
            return load_atd(meta, raw_path, n, PITCH, BLOCKS)

    def test_shapes_and_dtypes(self):
        t = self.load()
        self.assertEqual(t.ball_xy_m.shape, (N, 2))
        self.assertEqual(t.players_xy_m.shape, (N, 3, 2))
        self.assertEqual(t.ball_xy_m.dtype, np.float32)
        self.assertEqual(t.players_xy_m.dtype, np.float32)
        self.assertEqual(t.period.dtype, np.int8)
        self.assertIsNone(t.labels)
        self.assertEqual(t.pitch_m, PITCH)

    def test_slot_team_from_metadata(self):
        np.testing.assert_array_equal(self.load().slot_team, [HOME, AWAY, HOME])

    def test_metres_centred_and_frame_id_to_row(self):
        t = self.load()
        np.testing.assert_allclose(t.players_xy_m[0], [[0, 0], [-50, -30], [50, 30]])
        np.testing.assert_allclose(t.ball_xy_m[0], [-25, 15])
        np.testing.assert_allclose(t.players_xy_m[1, 0], [-40, -18], atol=1e-5)

    def test_nan_handling(self):
        t = self.load()
        self.assertTrue(np.isnan(t.ball_xy_m[1]).all())           # NaN ball
        self.assertTrue(np.isnan(t.players_xy_m[2, 1]).all())     # NaN player triple
        self.assertFalse(np.isnan(t.players_xy_m[2, [0, 2]]).any())
        self.assertTrue(np.isnan(t.players_xy_m[4, 0]).all())     # bare NaN group
        np.testing.assert_allclose(t.players_xy_m[4, 1], [-20, -6], atol=1e-5)
        for row in (3, 6, 7):                                     # absent frame_ids
            self.assertTrue(np.isnan(t.ball_xy_m[row]).all())
            self.assertTrue(np.isnan(t.players_xy_m[row]).all())

    def test_period_from_half_blocks(self):
        np.testing.assert_array_equal(self.load().period, [0, 1, 1, 0, 2, 2, 0, 0])

    def test_malformed_lines_fail_without_values(self):
        cases = {"1:0.1,0.2,0.0;0.3,0.4,0.0:0.5,0.5,0.0\n": "player groups",
                 "1:0.5,0.5,0.0\n": "fields",
                 "9:0.1,0.2,0;0.3,0.4,0;0.5,0.6,0:0.5,0.5,0\n": "frame_id outside"}
        for raw, message in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message) as ctx:
                self.load(raw)
            self.assertNotIn("0.5", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
