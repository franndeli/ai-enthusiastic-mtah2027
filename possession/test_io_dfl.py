import tempfile
import unittest
from pathlib import Path

import numpy as np

from possession.frames import AWAY, HOME, A, D, H, X
from possession.io_dfl import count_spells, load_dfl

MATCHINFO = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-TEST" HomeTeamId="T-H" GuestTeamId="T-G"/>
<Environment PitchX="105.00" PitchY="68.00"/>
<Teams>
<Team Role="guest" TeamId="T-G"><Players>
<Player PersonId="g1"/><Player PersonId="g2"/><Player PersonId="g-bench"/></Players></Team>
<Team Role="home" TeamId="T-H"><Players><Player PersonId="h1"/><Player PersonId="h2"/></Players></Team>
</Teams>
</MatchInformation></PutDataRequest>
"""


def frameset(section, team, person, frames):
    rows = "".join(f'<Frame N="{n}" X="{x}" Y="{y}"{extra}/>\n' for n, x, y, extra in frames)
    return f'<FrameSet GameSection="{section}" TeamId="{team}" PersonId="{person}">\n{rows}</FrameSet>\n'


def ball(status, possession):
    return f' BallStatus="{status}" BallPossession="{possession}"'


def positions():
    sets = []
    for section, base in (("firstHalf", 10000), ("secondHalf", 100000)):
        sets.append(frameset(section, "referee", "ref", [(base + i, 0, 0, "") for i in range(4)]))
        for k, (team, pid) in enumerate((("T-H", "h1"), ("T-H", "h2"), ("T-G", "g1"), ("T-G", "g2"))):
            # g2 only appears in the second half: NaN while not on the pitch.
            if pid == "g2" and section == "firstHalf":
                continue
            sets.append(frameset(section, team, pid, [(base + i, k + i, -k, "") for i in range(4)]))
    # First half: home ball, dead, away ball, then a frame with no ball record.
    sets.append(frameset("firstHalf", "BALL", "ball",
                         [(10000, 1.5, 2.5, ball(1, 1)), (10001, 0, 0, ball(0, 1)),
                          (10002, 0, 0, ball(1, 2))]))
    sets.append(frameset("secondHalf", "BALL", "ball",
                         [(100000 + i, 0, 0, ball(1, 2)) for i in range(4)]))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<PutDataRequest><Positions>\n'
            + "".join(sets) + "</Positions></PutDataRequest>\n")


class LoadDflTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            info, pos = Path(tmp) / "info.xml", Path(tmp) / "pos.xml"
            info.write_text(MATCHINFO, encoding="utf-8")
            pos.write_text(positions(), encoding="utf-8")
            cls.table = load_dfl(info, pos)

    def test_shapes_and_separator(self):
        t = self.table
        self.assertEqual(t.n_frames, 9)
        np.testing.assert_array_equal(t.period, [1, 1, 1, 1, 0, 2, 2, 2, 2])
        self.assertEqual(t.players_xy_m.shape, (9, 4, 2))
        self.assertEqual(t.players_xy_m.dtype, np.float32)
        self.assertEqual(t.labels[4], X)
        self.assertTrue(np.isnan(t.ball_xy_m[4]).all() and np.isnan(t.players_xy_m[4]).all())
        self.assertEqual((t.fps, t.pitch_m), (25.0, (105.0, 68.0)))

    def test_slots_exclude_referee_ball_and_untracked_players(self):
        np.testing.assert_array_equal(self.table.slot_team, [HOME, HOME, AWAY, AWAY])

    def test_raw_centred_coordinates(self):
        t = self.table
        np.testing.assert_allclose(t.ball_xy_m[0], [1.5, 2.5])
        np.testing.assert_allclose(t.players_xy_m[2, 1], [3, -1])  # h2 frame 2: x=k+i, y=-k
        np.testing.assert_allclose(t.players_xy_m[6, 3], [4, -3])

    def test_player_off_pitch_is_nan(self):
        self.assertTrue(np.isnan(self.table.players_xy_m[:4, 3]).all())

    def test_labels(self):
        # Missing ball record in frame 3 -> D and NaN ball.
        np.testing.assert_array_equal(self.table.labels, [H, D, A, D, X, A, A, A, A])
        self.assertTrue(np.isnan(self.table.ball_xy_m[3]).all())

    def test_spells_count_per_half(self):
        self.assertEqual(count_spells(self.table), 3)


if __name__ == "__main__":
    unittest.main()
