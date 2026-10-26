"""Single entry point that regenerates the submission from the match data (prints aggregates only).

    python run.py --out evaluation/results/<name>.csv              # v4 possession pipeline
    python run.py --out evaluation/results/<name>.csv --model config/possession_v3.json   # v3
    python run.py --out evaluation/results/<name>.csv --model config/possession_v5.json   # v5 (candidate)
    python run.py --out evaluation/results/check.csv --baseline 50  # official baseline, as a check
"""

import sys

from possession.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
