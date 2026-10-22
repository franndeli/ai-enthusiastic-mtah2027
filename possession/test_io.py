"""Synthetic ATD boundary, cache and runner failure tests; no match assets are read."""
import contextlib
import io as stdio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from possession import io
import run


class ParseTests(unittest.TestCase):
    def parse(self, text, n_frames=1):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.txt'
            path.write_text(text)
            return io.parse_atd(str(path), n_frames, 1)

    def test_valid_missing_and_optional_speed(self):
        players, ball = self.parse('1:0.1,0.2,N:0.3,0.4,2\n2:N:N\n', 2)
        np.testing.assert_allclose(players[0, 0, :2], [0.1, 0.2])
        self.assertTrue(np.isnan(players[1]).all())
        self.assertTrue(np.isnan(ball[1]).all())
        self.assertEqual(ball[0, 2], 2)

    def test_malformed_fields_never_echo_tokens(self):
        for line, field in [('SECRET:N:N', 'frame_id'),
                            ('1:SECRET,0.2:N', 'player_x'),
                            ('1:0.1,SECRET:N', 'player_y'),
                            ('1:0.1,0.2,SECRET:N', 'player_speed'),
                            ('1:N:SECRET,0.2', 'ball_x'),
                            ('1:N:0.1,SECRET', 'ball_y'),
                            ('1:N:0.1,0.2,SECRET', 'ball_speed')]:
            with self.subTest(field=field), self.assertRaises(ValueError) as caught:
                self.parse(line)
            message = str(caught.exception)
            self.assertNotIn('SECRET', message)
            self.assertIn(field, message)
            self.assertTrue(caught.exception.__suppress_context__)

    def test_missing_field_has_location(self):
        with self.assertRaisesRegex(ValueError, 'video frame index 0, slot index 0: invalid player_y'):
            self.parse('1:0.1:N')

    def test_duplicate_missing_and_malformed_records_rejected(self):
        for text, frames, reason in [('1:N:N\n1:N:N', 2, 'duplicate'),
                                     ('1:N:N', 2, 'missing'),
                                     ('SECRET', 1, 'field structure'),
                                     ('999:N:N', 1, 'out of range')]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.parse(text, frames)


class CacheTests(unittest.TestCase):
    def test_identity_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            meta, raw = base / 'meta.xml', base / 'raw.txt'
            meta.write_text('<root><Team id="h"/><Team id="a"/><Player teamId="h"/></root>')
            raw.write_text('1:0.1,0.2:0.3,0.4\n')
            with patch.object(io, 'require', return_value=[str(meta), str(raw)]), \
                    patch.object(io, 'source_frames', return_value=1), \
                    patch.object(io, 'half_video_blocks', return_value=[]), \
                    patch.object(io, 'work_dir', return_value=base):
                io.load_match()
                cache = next(base.glob('*.npz'))
                with patch.object(io, 'parse_atd', wraps=io.parse_atd) as parse:
                    io.load_match()
                    parse.assert_not_called()
                    np.savez(cache, players=np.zeros((2, 1, 3)), ball=np.zeros((2, 3)))
                    io.load_match()
                    self.assertEqual(parse.call_count, 1)
                    meta.write_text(meta.read_text() + '\n')
                    io.load_match()
                    self.assertEqual(parse.call_count, 2)
                    raw.write_text('1:0.2,0.2:0.3,0.4\n')
                    io.load_match()
                    self.assertEqual(parse.call_count, 3)
                    with patch.object(io, 'PARSER_VERSION', 3):
                        io.load_match()
                    self.assertEqual(parse.call_count, 4)

    def test_runner_suppresses_original_failure(self):
        error_output = stdio.StringIO()
        with patch('sys.argv', ['run.py', '--out', 'unused.csv']), \
                patch.object(run, 'load_config', side_effect=ValueError('SECRET')), \
                contextlib.redirect_stderr(error_output), self.assertRaises(SystemExit) as caught:
            run.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn('ValueError', error_output.getvalue())
        self.assertNotIn('SECRET', error_output.getvalue())
        self.assertNotIn('Traceback', error_output.getvalue())
