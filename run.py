"""Single entry point that regenerates the submission from the match data (prints aggregates only).

    python run.py --out evaluation/results/<name>.csv              # v5 possession pipeline
    python run.py --out evaluation/results/<name>.csv --model config/possession_v6.json   # v6:
        # v5 + PathCRF ball-free team evidence (needs requirements-v6.txt; a few minutes on CPU)
    python run.py --out evaluation/results/<name>.csv --model config/possession_v4.json   # v4
    python run.py --out evaluation/results/<name>.csv --model config/possession_v3.json   # v3
    python run.py --out evaluation/results/check.csv --baseline 50  # official baseline, as a check
"""

import sys

from possession.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
