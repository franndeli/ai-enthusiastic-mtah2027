"""Tests for scripts/agent_guard.py. Stdlib only: python3 -m unittest discover -s scripts -p 'test_*.py'"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_guard  # noqa: E402

RULES = agent_guard.load_rules()
GUARD = Path(__file__).resolve().parent / "agent_guard.py"


def blocked(tool, **tool_input):
    return agent_guard.check(tool, tool_input, RULES) is not None


class BlocksDataAccess(unittest.TestCase):
    def test_blocks_reading_private_data_root(self):
        self.assertTrue(blocked("read", path="~/.mtah2027/data/tactical/BAR-ATH_tactical_atd.txt"))
        self.assertTrue(blocked("Read", file_path="/Users/x/.mtah2027/work/features.parquet"))

    def test_blocks_shell_access_to_data(self):
        self.assertTrue(blocked("bash", command="head -5 data/tactical/x.txt"))
        self.assertTrue(blocked("bash", command='python -c "open($MTAH_DATA_DIR)"'))
        self.assertTrue(blocked("Bash", command="ffprobe clip.mp4"))
        self.assertTrue(blocked("bash", command="grep foo BAR-ATH_tactical_homography.csv"))
        self.assertTrue(blocked("bash", command="cat /Users/x/Developer/mtah2027/data/tactical"))

    def test_blocks_frame_level_outputs(self):
        self.assertTrue(blocked("read", path="evaluation/results/my_run.csv"))
        self.assertTrue(blocked("bash", command="head evaluation/reports/x.csv"))

    def test_blocks_delegation_pointed_at_data(self):
        self.assertTrue(blocked("bg_delegate", prompt="inspect ~/.mtah2027/data and summarise"))
        self.assertTrue(blocked("codemode", code="await tools.read({path:'data/tactical/a.txt'})"))

    def test_blocks_writing_into_data(self):
        self.assertTrue(blocked("write", path="data/tactical/notes.md", content="x"))


class AllowsNormalWork(unittest.TestCase):
    def test_allows_code_and_docs(self):
        self.assertFalse(blocked("read", path="starter/baseline.py"))
        self.assertFalse(blocked("read", path="data/README.md"))
        self.assertFalse(blocked("read", path="/Users/x/Developer/mtah2027/DATA.md"))
        self.assertFalse(blocked("read", path="evaluation/results/baseline_smooth50.csv"))
        self.assertFalse(blocked("read", path="evaluation/gt/BAR-ATH_possession_gt_public.csv"))

    def test_judges_writes_by_path_not_prose(self):
        self.assertFalse(blocked("write", path="docs/PLAN.md", content="put files in data/tactical/ and .mp4"))
        self.assertFalse(blocked("edit", path="src/possession.py", edits=[{"oldText": "a", "newText": "data/x"}]))

    def test_allows_runners_that_print_aggregates(self):
        self.assertFalse(blocked("bash", command="scripts/py starter/baseline.py"))
        self.assertFalse(blocked("bash", command="scripts/data-status.sh"))
        self.assertFalse(blocked("bash", command="scripts/py evaluation/validate_submission.py evaluation/results/v1.csv"))
        self.assertFalse(blocked("bash", command="python evaluation/run_evaluation.py"))
        self.assertFalse(blocked("bash", command="scripts/py run.py --out evaluation/results/v1.csv"))
        self.assertFalse(blocked("bash", command="cd /Users/x/mtah2027 && scripts/py scripts/audit_data.py --half 1"))
        self.assertTrue(blocked("bash", command="scripts/py run.py --out evaluation/results/v1.csv && head evaluation/results/v1.csv"))
        self.assertTrue(blocked("bash", command="scripts/py scripts/x.py; cat data/tactical/a.txt"))

    def test_allowlist_rejects_chained_commands(self):
        self.assertTrue(blocked("bash", command="python evaluation/run_evaluation.py; head evaluation/results/v1.csv"))

    def test_repo_folder_name_is_not_the_private_root(self):
        self.assertFalse(blocked("bash", command="ls ~/Developer/mtah2027/starter"))


class HookProtocol(unittest.TestCase):
    def run_guard(self, payload):
        return subprocess.run([sys.executable, str(GUARD)], input=payload, capture_output=True, text=True)

    def test_exit_codes(self):
        ok = self.run_guard(json.dumps({"tool_name": "read", "tool_input": {"path": "README.md"}}))
        self.assertEqual(ok.returncode, 0)
        no = self.run_guard(json.dumps({"tool_name": "read", "tool_input": {"path": "data/tactical/a"}}))
        self.assertEqual(no.returncode, 2)
        self.assertIn("DATA GUARD", no.stderr)

    def test_fails_closed_on_bad_input(self):
        self.assertEqual(self.run_guard("not json").returncode, 2)


if __name__ == "__main__":
    unittest.main()
