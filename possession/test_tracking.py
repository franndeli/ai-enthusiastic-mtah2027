import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.tracking import read_epts  # noqa: E402

META = """<?xml version="1.0" encoding="utf-8"?>
<main><Metadata><GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>
<Teams><Team id="T1"><Name>Home</Name></Team><Team id="T2"><Name>Away</Name></Team></Teams>
<Players>
<Player id="P1" teamId="T1"><ShirtNumber>1</ShirtNumber></Player>
<Player id="P2" teamId="T2"><ShirtNumber>2</ShirtNumber></Player>
<Player id="P3" teamId="T1"><ShirtNumber>3</ShirtNumber></Player>
</Players></Metadata></main>
"""
# DATA.md format: "<frame>:<player>;<player>;...:<ball>", groups "x,y,speed", NaN when missing
RAW = ("1:0.5,0.5,1.0;0.25,0.75,2.0;NaN,NaN,NaN:0.5,0.5,0.0\n"
       "2:0.5,0.5,1.0;NaN,NaN,NaN;0.1,0.2,3.0:NaN,NaN,NaN\n"
       "3:1.0,0.0,0.0;0.0,1.0,0.0;0.5,0.5,0.0:0.6,0.4,0.1\n")


class TestReader(unittest.TestCase):
    def test_atd_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta, raw = os.path.join(tmp, "m.xml"), os.path.join(tmp, "r.txt")
            open(meta, "w").write(META)
            open(raw, "w").write(RAW)
            trk = read_epts(meta, raw, pitch=(105.3, 68.0), n_frames=3)
        self.assertEqual(trk.players.shape, (3, 3, 2))
        self.assertEqual(trk.team.tolist(), [0, 1, 0])           # first <Team> is home
        np.testing.assert_allclose(trk.players[0, 0], [52.65, 34.0], atol=1e-4)
        np.testing.assert_allclose(trk.players[0, 1], [26.325, 51.0], atol=1e-4)
        self.assertTrue(np.isnan(trk.players[0, 2]).all())
        self.assertTrue(np.isnan(trk.ball[1]).all())             # frame_id 2 -> row 1
        np.testing.assert_allclose(trk.ball[2], [63.18, 27.2], atol=1e-4)
        np.testing.assert_allclose(trk.extra["player_speed"][1, 2], 3.0)


if __name__ == "__main__":
    unittest.main()
