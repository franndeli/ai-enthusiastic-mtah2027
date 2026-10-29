#!/usr/bin/env python3
"""Regenerate the submission cold (re-parsed tracking) and warm (cached arrays), check that both
are byte-identical and valid, and print the public scores. Aggregates and hashes only.

    scripts/py scripts/verify_submission.py --name ai-enthusiastic [--config config/possession.json]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".conda-env" / "bin" / "python")


def run(args: list[str]) -> str:
    started = time.time()
    proc = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("step failed: {} ({:.0f}s)".format(args[0], time.time() - started))
    return "{:.1f}s".format(time.time() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "possession.json"))
    args = parser.parse_args()
    final = ROOT / "evaluation" / "results" / f"{args.name}.csv"
    work = Path(os.environ.get("MTAH_WORK_DIR", tempfile.gettempdir()))
    cold = work / f"{args.name}.cold.csv"
    t_cold = run(["run.py", "--no-cache", "--config", args.config, "--out", str(cold)])
    t_warm = run(["run.py", "--config", args.config, "--out", str(final)])
    h_cold = hashlib.sha256(cold.read_bytes()).hexdigest()
    h_warm = hashlib.sha256(final.read_bytes()).hexdigest()
    cold.unlink()
    print("cold {}  warm {}  sha256 {}  {}".format(
        t_cold, t_warm, h_warm[:16], "IDENTICAL" if h_cold == h_warm else "DIFFERENT"))
    validator = subprocess.run([PY, "evaluation/validate_submission.py", str(final)], cwd=ROOT,
                               capture_output=True, text=True)
    print("validator exit {}".format(validator.returncode))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "evaluation"))
    import possession_metrics as pm  # noqa: E402
    from possession import io  # noqa: E402
    result = pm.evaluate(io.load_public_gt(), pm.label_codes(pm.read_label_csv(str(final))),
                         bootstrap_n=1000)
    print("public spellF1 {:.4f} [{:.4f}, {:.4f}]  macroF1 {:.4f}  possessions {} / {}".format(
        result["spell_f1_mean"], result["spell_bootstrap_low"], result["spell_bootstrap_high"],
        result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]))
    return 0 if h_cold == h_warm and validator.returncode == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"verify_submission failed: {type(exc).__name__} (details suppressed)", file=sys.stderr)
        raise SystemExit(1)
