"""Compare two frame,label CSVs label by label (ignores CRLF vs LF). Exit 0 if identical.

usage: python scripts/compare_csv.py a.csv b.csv
"""
import sys

import pandas as pd

a = pd.read_csv(sys.argv[1], dtype={"label": "object"})
b = pd.read_csv(sys.argv[2], dtype={"label": "object"})
same = len(a) == len(b) and (a["frame"].values == b["frame"].values).all() and \
    (a["label"].str.strip().values == b["label"].str.strip().values).all()
if same:
    print("IDENTICAL: %d rows" % len(a))
else:
    n = min(len(a), len(b))
    diff = (a["label"].values[:n] != b["label"].values[:n]).sum()
    print("DIFFERENT: rows %d vs %d, %d labels differ" % (len(a), len(b), diff))
sys.exit(0 if same else 1)
