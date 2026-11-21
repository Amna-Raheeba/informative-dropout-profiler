"""Notebook 02 - the headline evaluation experiment.

Runs the full seeded evaluation (6 paired seeds x 3 regimes x
4 imputers x 4 predictors) and prints the summary: illusion gaps,
imputation-ranking reversal, diagnosis accuracy against the two
baselines, and driver identification. Runnable as a plain script:
python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dropout_profiler.evaluate import headline_numbers, run_evaluation
from dropout_profiler.report import render_evaluation_summary


def main():
    evaluation = run_evaluation()
    print(render_evaluation_summary(evaluation, headline_numbers(evaluation)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
