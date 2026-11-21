"""Notebook 01 - demo run.

Runs the full predictor x imputer grid on one paired seed (seed 7)
under all three dropout regimes and prints the profiler's diagnosis
for each regime next to the planted truth, plus the RMSE grids.
Runnable as a plain script: python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dropout_profiler.evaluate import run_single_seed
from dropout_profiler.report import render_single_seed_report


def main():
    result = run_single_seed(7)
    print(render_single_seed_report(result))
    correct = all(
        result["regimes"][regime]["profile"]["diagnosis"] == regime
        for regime in ("iid", "grouped", "informative")
    )
    print("Diagnosis exact on all three regimes: %s" % correct)
    return 0


if __name__ == "__main__":
    sys.exit(main())
