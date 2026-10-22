"""Possession labelling for MTAH 2027 Phase 1: pure modules over numpy arrays, I/O kept in io.py.

Pipeline (run.py): io.load_match -> features.compute -> evidence.scores -> decoder.decode
-> structure.decode_halves -> io.write_submission. Each stage is a pure function over arrays so it can be
unit-tested on synthetic input and swapped independently.
"""

H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
FPS = 25
