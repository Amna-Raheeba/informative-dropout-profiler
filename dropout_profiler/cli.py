"""Command-line interface.

Usage:
    python -m dropout_profiler demo
    python -m dropout_profiler evaluate --out results_dir
"""

import argparse
import csv
import json
import os
import sys

from .evaluate import headline_numbers, run_evaluation, run_single_seed
from .models import IMPUTERS, PREDICTORS, REGIMES
from .report import render_evaluation_summary, render_single_seed_report


def _jsonable(obj):
    if isinstance(obj, dict):
        return dict((k, _jsonable(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def cmd_demo(args):
    result = run_single_seed(7)
    sys.stdout.write(render_single_seed_report(result))
    return 0


def cmd_evaluate(args):
    evaluation = run_evaluation()
    headline = headline_numbers(evaluation)
    out_dir = args.out
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    slim = {
        "seeds": evaluation["seeds"],
        "n_runs": evaluation["n_runs"],
        "missing_rate": evaluation["missing_rate"],
        "rmse_mean": evaluation["rmse_mean"],
        "complete_rmse_mean": evaluation["complete_rmse_mean"],
        "diagnosis_by_regime": evaluation["diagnosis_by_regime"],
        "cell_gaps": evaluation["cell_gaps"],
        "worst_gap_cell": evaluation["worst_gap_cell"],
        "headline": headline,
    }
    with open(os.path.join(out_dir, "evaluation.json"), "w") as fh:
        json.dump(_jsonable(slim), fh, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "evaluation.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["regime", "imputer", "predictor", "mean_rmse"])
        for regime in REGIMES:
            for imputer in IMPUTERS:
                for predictor in PREDICTORS:
                    writer.writerow(
                        [
                            regime,
                            imputer,
                            predictor,
                            "%.4f"
                            % evaluation["rmse_mean"][regime][imputer][predictor],
                        ]
                    )
        for predictor in PREDICTORS:
            writer.writerow(
                [
                    "complete",
                    "none",
                    predictor,
                    "%.4f" % evaluation["complete_rmse_mean"][predictor],
                ]
            )
    with open(os.path.join(out_dir, "evaluation_summary.md"), "w") as fh:
        fh.write(render_evaluation_summary(evaluation, headline))
    with open(os.path.join(out_dir, "HEADLINE.md"), "w") as fh:
        fh.write("# Headline results\n\n")
        fh.write(
            "Across %d paired seeds at matched missing rates "
            "(iid %.3f, grouped %.3f, informative %.3f), the "
            "configuration a standard iid benchmark would choose "
            "degrades by %+.2f days RMSE under grouped dropout and "
            "%+.2f days under informative dropout (%+.2f vs complete "
            "data). The illusion concentrates in the naive fills: "
            "under informative dropout the mean-fill cells degrade "
            "by %+.2f days on average and the worst cell (%s) by "
            "%+.2f days, while the same fills barely move under "
            "grouped dropout. The best imputer under iid is not the "
            "best under "
            "informative dropout in %.1f%% of (seed, predictor) pairs "
            "and is outright the worst in %.1f%%. The profiler names "
            "the planted regime with accuracy %.3f (MAR screen %.3f, "
            "always-iid %.3f) and identifies the dropout driver with "
            "accuracy %.3f.\n"
            % (
                evaluation["n_runs"],
                headline["rate_iid"],
                headline["rate_grouped"],
                headline["rate_informative"],
                headline["gap_grouped"],
                headline["gap_informative"],
                headline["gap_informative_vs_complete"],
                headline["gap_informative_mean_fill"],
                evaluation["worst_gap_cell"],
                headline["worst_gap_informative"],
                100.0 * evaluation["reversal_rate_informative"],
                100.0 * evaluation["iid_best_is_worst_rate"],
                headline["diagnosis_accuracy"],
                headline["mar_screen_accuracy"],
                headline["always_iid_accuracy"],
                headline["driver_accuracy"],
            )
        )
    sys.stdout.write("Wrote evaluation results to %s\n" % out_dir)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="dropout_profiler")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "evaluate":
        return cmd_evaluate(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
