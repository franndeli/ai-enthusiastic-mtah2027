"""Single entry point that regenerates the submission from the match data (prints aggregates only).

    python run.py --out evaluation/results/<name>.csv              # v6 (default): v5 + PathCRF
        # ball-free team evidence; needs requirements-v6.txt, a few minutes on CPU
    python run.py --out evaluation/results/<name>.csv --model config/possession_v5.json   # v5
    python run.py --out evaluation/results/<name>.csv --model config/possession_v4.json   # v4
    python run.py --out evaluation/results/<name>.csv --model config/possession_v3.json   # v3
    python run.py --out evaluation/results/check.csv --baseline 50  # official baseline, as a check
    python scripts/check_v6.py      # runs v6 and compares it with the team's reference (PASS/FAIL)
"""

import sys

from possession.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
