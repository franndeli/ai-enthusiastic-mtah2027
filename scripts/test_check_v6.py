import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_v6  # noqa: E402


class TestCheckV6(unittest.TestCase):
    def test_read_pins_skips_comments_and_options(self):
        text = ("# comment\n--index-url https://example.org\nnumpy==2.4.6\n\n"
                "torch_geometric==2.8.1  # pinned\nscipy>=1\n")
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "req.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            self.assertEqual(check_v6.read_pins(path),
                             {"numpy": "2.4.6", "torch_geometric": "2.8.1"})

    def test_compare_ignores_build_labels(self):
        pins = {"torch": "2.14.1", "numpy": "2.4.6", "pandas": "3.0.6"}
        found = {"torch": "2.14.1+cpu", "numpy": "2.4.5", "pandas": None}
        self.assertEqual(check_v6.compare(pins, found),
                         {"torch": "ok", "numpy": "different", "pandas": "missing"})

    def test_shipped_pins_and_reference(self):
        pins = check_v6.read_pins()
        for name in ("numpy", "pandas", "torch", "torch_geometric", "scipy"):
            self.assertIn(name, pins)
        with open(check_v6.REFERENCE, encoding="utf-8") as handle:
            ref = json.load(handle)["BAR-ATH"][check_v6.MODEL]
        self.assertRegex(ref["sha256"], re.compile(r"^[0-9a-f]{64}$"))


if __name__ == "__main__":
    unittest.main()
