"""Plain-text / markdown renderers for profiles and evaluations."""

from .models import IMPUTERS, PREDICTORS, REGIMES


def render_single_seed_report(result):
    lines = []
    lines.append("Informative Dropout Profiler - demo fleet (seed %d)" % result["seed"])
    lines.append(
        "Train units: %d, test units: %d. Task: RUL regression at "
        "landmark days." % (result["n_train"], result["n_test"])
    )
    lines.append("")
    lines.append("Complete-data reference RMSE (no dropout):")
    for predictor in PREDICTORS:
        lines.append(
            "  %-14s %.2f days" % (predictor, result["complete_rmse"][predictor])
        )
    for regime in REGIMES:
        block = result["regimes"][regime]
        profile = block["profile"]
        lines.append("")
        lines.append("Regime: %s (planted driver: %s)" % (regime, block["truth"]["driver"]))
        lines.append(
            "  achieved missing rate %.3f | diagnosis: %s (driver: %s)"
            % (profile["missing_rate"], profile["diagnosis"], profile["driver"])
        )
        lines.append(
            "  same-group lift %.2f | cross-group lift %.2f | "
            "proxy gradient %+.3f | life gradient %+.3f | "
            "mean run length %.2f"
            % (
                profile["same_group_lift"],
                profile["cross_group_lift"],
                profile["proxy_gradient"],
                profile["life_gradient"],
                profile["mean_run_length"],
            )
        )
        lines.append(
            "  worst group: %s | top gradient sensors: %s"
            % (profile["worst_group"], ", ".join(profile["top_gradient_sensors"]))
        )
        lines.append("  RMSE by imputer x predictor:")
        lines.append("  %-10s %s" % ("imputer", "  ".join("%-13s" % p for p in PREDICTORS)))
        for imputer in IMPUTERS:
            row = block["rmse"][imputer]
            lines.append(
                "  %-10s %s"
                % (imputer, "  ".join("%-13.2f" % row[p] for p in PREDICTORS))
            )
    lines.append("")
    return "\n".join(lines)


def render_evaluation_summary(evaluation, headline):
    lines = []
    lines.append("# Evaluation summary")
    lines.append("")
    lines.append(
        "%d paired seeds (%s); train fleets of %d units, test fleets "
        "of %d units. Every regime masks the same machines - only the "
        "missingness mechanism changes."
        % (
            evaluation["n_runs"],
            ", ".join(str(s) for s in evaluation["seeds"]),
            evaluation["runs"][0]["n_train"],
            evaluation["runs"][0]["n_test"],
        )
    )
    lines.append("")
    lines.append("## Achieved missing rates (calibrated to match)")
    lines.append("")
    for regime in REGIMES:
        lines.append(
            "- %s: %.3f" % (regime, evaluation["missing_rate"][regime])
        )
    lines.append("")
    lines.append("## Robustness illusion gap")
    lines.append("")
    lines.append(
        "Configuration chosen under iid dropout (best imputer per "
        "predictor), re-scored under the other regimes. Gap = RMSE "
        "under the regime minus RMSE under iid, in days."
    )
    lines.append("")
    lines.append("- grouped gap: %+.2f days" % headline["gap_grouped"])
    lines.append("- informative gap: %+.2f days" % headline["gap_informative"])
    lines.append(
        "- informative vs complete data: %+.2f days"
        % headline["gap_informative_vs_complete"]
    )
    lines.append("")
    lines.append(
        "Per-cell gaps (regime RMSE minus iid RMSE, days) show where "
        "the illusion lives: it concentrates in the naive fills. "
        "Under informative dropout the mean-fill gap averages %+.2f "
        "days across predictors and the unit-mean gap %+.2f, while "
        "the worst cell (%s) degrades by %+.2f days. Under grouped "
        "dropout the same fills barely move - the damage is "
        "mechanism-specific, and an iid benchmark cannot see it."
        % (
            headline["gap_informative_mean_fill"],
            headline["gap_informative_unit_mean"],
            evaluation["worst_gap_cell"],
            headline["worst_gap_informative"],
        )
    )
    lines.append("")
    lines.append("## Imputation-ranking reversal")
    lines.append("")
    lines.append(
        "- best imputer changes iid -> informative in %.1f%% of "
        "(seed, predictor) pairs"
        % (100.0 * evaluation["reversal_rate_informative"])
    )
    lines.append(
        "- best imputer changes iid -> grouped in %.1f%% of pairs"
        % (100.0 * evaluation["reversal_rate_grouped"])
    )
    lines.append(
        "- the iid-best imputer is the *worst* under informative "
        "dropout in %.1f%% of pairs"
        % (100.0 * evaluation["iid_best_is_worst_rate"])
    )
    lines.append("")
    lines.append("## Diagnosis and drivers")
    lines.append("")
    lines.append(
        "- profiler diagnosis accuracy: %.3f (iid %.2f, grouped %.2f, "
        "informative %.2f)"
        % (
            headline["diagnosis_accuracy"],
            evaluation["diagnosis_by_regime"]["iid"],
            evaluation["diagnosis_by_regime"]["grouped"],
            evaluation["diagnosis_by_regime"]["informative"],
        )
    )
    lines.append(
        "- MAR-screen baseline accuracy: %.3f" % headline["mar_screen_accuracy"]
    )
    lines.append(
        "- always-iid baseline accuracy: %.3f" % headline["always_iid_accuracy"]
    )
    lines.append(
        "- driver identification accuracy: %.3f" % headline["driver_accuracy"]
    )
    lines.append("")
    lines.append("## Mean RMSE by regime x imputer x predictor")
    lines.append("")
    for regime in REGIMES:
        lines.append("### %s" % regime)
        lines.append("")
        lines.append(
            "| imputer | %s |" % " | ".join(PREDICTORS)
        )
        lines.append("|---|%s|" % "|".join("---" for _ in PREDICTORS))
        for imputer in IMPUTERS:
            row = evaluation["rmse_mean"][regime][imputer]
            lines.append(
                "| %s | %s |"
                % (imputer, " | ".join("%.2f" % row[p] for p in PREDICTORS))
            )
        lines.append("")
    lines.append("### complete data (reference)")
    lines.append("")
    for predictor in PREDICTORS:
        lines.append(
            "- %s: %.2f days"
            % (predictor, evaluation["complete_rmse_mean"][predictor])
        )
    lines.append("")
    return "\n".join(lines)
