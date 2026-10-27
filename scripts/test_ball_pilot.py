import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ball_detector_pilot as pilot  # noqa: E402


class TestBallPilot(unittest.TestCase):
    def test_tiles_cover_the_frame(self):
        self.assertEqual(pilot.tiles(1920, 1080, 1280), [(0, 0), (640, 0)])
        self.assertEqual(pilot.tiles(1280, 720, 1280), [(0, 0)])
        self.assertEqual(pilot.tiles(1920, 1080, 0), [(0, 0)])
        corners = pilot.tiles(3840, 2160, 1280)
        covered = np.zeros((2160, 3840), dtype=bool)
        for x, y in corners:
            covered[y:y + 1280, x:x + 1280] = True
        self.assertTrue(covered.all())

    def test_dedupe_keeps_the_most_confident(self):
        out = pilot.dedupe([(100, 100, 0.2), (103, 101, 0.6), (400, 300, 0.4)])
        self.assertEqual(out, [(103, 101, 0.6), (400, 300, 0.4)])

    def test_to_pitch_inverts_the_homography(self):
        H = np.array([[1000.0, 50.0, 100.0], [10.0, 600.0, 80.0], [0.0, 0.1, 1.0]])
        x, y = 0.3, 0.6                                     # normalised pitch
        px = H @ np.array([x, y, 1.0])
        dets = [(px[0] / px[2], px[1] / px[2], 0.9), (-5000.0, -5000.0, 0.8)]
        out = pilot.to_pitch(dets, H, (105.3, 68.0))
        self.assertEqual(len(out), 1)                       # the second one is off the pitch
        self.assertAlmostEqual(out[0][0], x * 105.3, places=6)
        self.assertAlmostEqual(out[0][1], y * 68.0, places=6)
        self.assertEqual(pilot.to_pitch(dets, np.full((3, 3), np.nan), (105.3, 68.0)), [])

    def test_alignment_finds_the_offset(self):
        n = 400
        t = np.arange(n, dtype=float)
        atd = np.stack([10 + 0.3 * t, 30 + 5 * np.sin(t / 15)], axis=1)
        frames = np.arange(50, 300)
        best = atd[frames + 2]                              # detections lag the ATD by 2 frames
        top = min(pilot.alignment(frames, best, atd), key=lambda r: r[1])
        self.assertEqual(top[0], 2)
        self.assertLess(top[1], 1e-9)

    def test_summary_counts(self):
        k = 100
        frames = np.arange(1000, 1000 + k)
        atd = np.full((k, 2), 20.0)
        atd[60:] = np.nan
        best = np.full((k, 2), 20.5)
        best[80:] = np.nan
        bucket = np.where(np.isnan(atd[:, 0]), 2, 0)
        players = np.full((k, 46, 2), np.nan)
        players[:, 0] = 21.0
        lines = "\n".join(pilot.summarise(frames, best, np.full(k, 0.5), atd, bucket, players,
                                          25.0, 1))
        self.assertIn("ATD 60.0 %, detector 80.0 %, both 60.0 %, either 80.0 %", lines)
        self.assertIn("within 1 m 100.0 %", lines)
        self.assertIn("gaps 1-10 s", lines)
        self.assertIn("detector fills 50.0 %", lines)


if __name__ == "__main__":
    unittest.main()
