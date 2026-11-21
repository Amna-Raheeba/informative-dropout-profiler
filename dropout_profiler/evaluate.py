"""The headline experiment: predictors x imputers x dropout regimes.

For each seed, one paired fleet is generated once and masked under
all three regimes (plus a complete-data reference). A separate test
fleet (seed + TEST_SEED_OFFSET) is masked the same way. Every
(predictor, imputer) pair is trained on the masked training fleet and
scored on the masked test fleet, so each cell of the grid is an
honest held-out RUL RMSE under that missingness mechanism.

Headline objects:

- **Robustness illusion gap.** For each predictor, take the imputer
  that wins under iid dropout (what a standard benchmark would
  select) and measure how much worse that chosen configuration scores
  under grouped and under informative dropout. The gap is the part of
  the benchmark score that was an artifact of benchmarking against
  the wrong missingness.
- **Imputation-ranking reversal rate.** How often the best imputer
  under iid is not the best imputer under informative dropout - and
  how often it is outright the worst.
- **Diagnosis accuracy.** The profiler names the regime from the
  masked training fleet alone; scored against the planted truth,
  against an always-iid baseline (the standard assumption) and
  against an observed-covariate MAR screen.
- **Driver identification.** Grouped: does the profiler name bus_b
  as the worst group? Informative: are its top-2 gradient sensors
  exactly the two most vulnerable sensors?
"""

from .features import build_rows, sensor_means
from .generate import apply_dropout, generate_fleet
from .models import (
    IMPUTERS,
    PREDICTORS,
    REGIMES,
    MaskedUnit,
    SENSOR_NAMES,
)
from .predictors import fit_predictor, rmse
from .profile import diagnose_mar_screen, profile_fleet

EVAL_SEEDS = (201, 202, 203, 204, 205, 206)
TEST_SEED_OFFSET = 500000
FLEET_TRAIN = 36
FLEET_TEST = 20


def _as_complete(units):
    """Wrap complete units as masked units with nothing missing."""
    wrapped = []
    for unit in units:
        masked = dict((name, list(unit.values[name])) for name in SENSOR_NAMES)
        mask = dict(
            (name, [False] * unit.lifetime) for name in SENSOR_NAMES
        )
        wrapped.append(MaskedUnit(unit, masked, mask))
    return wrapped


def _score_grid(train_units, test_units):
    """RMSE for every (imputer, predictor) on one train/test pair."""
    means = sensor_means(train_units)
    scores = {}
    for imputer in IMPUTERS:
        X_train, y_train = build_rows(train_units, imputer, means)
        X_test, y_test = build_rows(test_units, imputer, means)
        scores[imputer] = {}
        for predictor in PREDICTORS:
            model = fit_predictor(predictor, X_train, y_train)
            scores[imputer][predictor] = rmse(model.predict(X_test), y_test)
    return scores


def _score_complete(train_units, test_units):
    means = sensor_means(train_units)
    X_train, y_train = build_rows(train_units, "mean", means)
    X_test, y_test = build_rows(test_units, "mean", means)
    scores = {}
    for predictor in PREDICTORS:
        model = fit_predictor(predictor, X_train, y_train)
        scores[predictor] = rmse(model.predict(X_test), y_test)
    return scores


def run_single_seed(seed, n_train=FLEET_TRAIN, n_test=FLEET_TEST):
    """Full grid + profile for one paired seed."""
    train_complete = generate_fleet(seed, n_train)
    test_complete = generate_fleet(seed + TEST_SEED_OFFSET, n_test)
    complete_rmse = _score_complete(
        _as_complete(train_complete), _as_complete(test_complete)
    )
    regimes = {}
    for regime in REGIMES:
        masked_train, truth = apply_dropout(train_complete, regime, seed)
        masked_test, _ = apply_dropout(
            test_complete, regime, seed + TEST_SEED_OFFSET
        )
        profile = profile_fleet(masked_train)
        regimes[regime] = {
            "truth": truth,
            "profile": profile,
            "rmse": _score_grid(masked_train, masked_test),
        }
    return {
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "complete_rmse": complete_rmse,
        "regimes": regimes,
    }


def _argmin_imputer(scores, predictor):
    return min(IMPUTERS, key=lambda imp: scores[imp][predictor])


def _argmax_imputer(scores, predictor):
    return max(IMPUTERS, key=lambda imp: scores[imp][predictor])


def run_evaluation(seeds=EVAL_SEEDS, n_train=FLEET_TRAIN, n_test=FLEET_TEST):
    """Aggregate the grid over seeds into the headline measurements."""
    runs = [run_single_seed(seed, n_train, n_test) for seed in seeds]
    n = len(runs)

    missing_rate = {}
    for regime in REGIMES:
        missing_rate[regime] = sum(
            run["regimes"][regime]["truth"]["achieved_rate"] for run in runs
        ) / n

    rmse_mean = {}
    for regime in REGIMES:
        rmse_mean[regime] = dict(
            (
                imputer,
                dict(
                    (
                        predictor,
                        sum(
                            run["regimes"][regime]["rmse"][imputer][predictor]
                            for run in runs
                        ) / n,
                    )
                    for predictor in PREDICTORS
                ),
            )
            for imputer in IMPUTERS
        )
    complete_mean = dict(
        (
            predictor,
            sum(run["complete_rmse"][predictor] for run in runs) / n,
        )
        for predictor in PREDICTORS
    )

    gaps = {"grouped": [], "informative": []}
    gaps_vs_complete = []
    reversals = {"grouped": 0, "informative": 0}
    iid_best_is_worst = 0
    pairs = 0
    chosen = {}
    for run in runs:
        for predictor in PREDICTORS:
            pairs += 1
            picked = _argmin_imputer(run["regimes"]["iid"]["rmse"], predictor)
            chosen.setdefault(predictor, []).append(picked)
            base = run["regimes"]["iid"]["rmse"][picked][predictor]
            for regime in ("grouped", "informative"):
                gaps[regime].append(
                    run["regimes"][regime]["rmse"][picked][predictor] - base
                )
                if _argmin_imputer(
                    run["regimes"][regime]["rmse"], predictor
                ) != picked:
                    reversals[regime] += 1
            gaps_vs_complete.append(
                run["regimes"]["informative"]["rmse"][picked][predictor]
                - run["complete_rmse"][predictor]
            )
            if _argmax_imputer(
                run["regimes"]["informative"]["rmse"], predictor
            ) == picked:
                iid_best_is_worst += 1

    # Per-cell illusion gaps from the mean grid: for every
    # (imputer, predictor) cell, RMSE under the regime minus RMSE
    # under iid. The illusion is imputer-dependent - it concentrates
    # in the naive fills - so it is reported per cell, not only for
    # the iid-chosen configuration.
    cell_gaps = {}
    for regime in ("grouped", "informative"):
        cell_gaps[regime] = dict(
            (
                imputer,
                dict(
                    (
                        predictor,
                        rmse_mean[regime][imputer][predictor]
                        - rmse_mean["iid"][imputer][predictor],
                    )
                    for predictor in PREDICTORS
                ),
            )
            for imputer in IMPUTERS
        )
    worst_gap = None
    worst_cell = None
    for imputer in IMPUTERS:
        for predictor in PREDICTORS:
            gap = cell_gaps["informative"][imputer][predictor]
            if worst_gap is None or gap > worst_gap:
                worst_gap = gap
                worst_cell = "%s/%s" % (imputer, predictor)
    mean_fill_gap = sum(
        cell_gaps["informative"]["mean"][p] for p in PREDICTORS
    ) / len(PREDICTORS)
    unit_mean_gap = sum(
        cell_gaps["informative"]["unit_mean"][p] for p in PREDICTORS
    ) / len(PREDICTORS)

    diagnosis_hits = 0
    mar_hits = 0
    driver_hits = 0
    cases = 0
    per_regime_diag = {}
    for regime in REGIMES:
        hits = 0
        for run in runs:
            profile = run["regimes"][regime]["profile"]
            cases += 1
            if profile["diagnosis"] == regime:
                diagnosis_hits += 1
                hits += 1
            if diagnose_mar_screen(profile) == regime:
                mar_hits += 1
            if regime == "grouped":
                if profile["worst_group"] == "bus_b":
                    driver_hits += 1
            elif regime == "informative":
                if set(profile["top_gradient_sensors"]) == {
                    "vibration",
                    "temperature",
                }:
                    driver_hits += 1
            else:
                if profile["diagnosis"] == "iid":
                    driver_hits += 1
        per_regime_diag[regime] = hits / float(n)

    return {
        "seeds": list(seeds),
        "n_runs": n,
        "missing_rate": missing_rate,
        "rmse_mean": rmse_mean,
        "complete_rmse_mean": complete_mean,
        "chosen_imputers": chosen,
        "gap_grouped": sum(gaps["grouped"]) / len(gaps["grouped"]),
        "gap_informative": sum(gaps["informative"]) / len(gaps["informative"]),
        "gap_informative_vs_complete": (
            sum(gaps_vs_complete) / len(gaps_vs_complete)
        ),
        "cell_gaps": cell_gaps,
        "gap_informative_mean_fill": mean_fill_gap,
        "gap_informative_unit_mean": unit_mean_gap,
        "worst_gap_informative": worst_gap,
        "worst_gap_cell": worst_cell,
        "reversal_rate_grouped": reversals["grouped"] / float(pairs),
        "reversal_rate_informative": reversals["informative"] / float(pairs),
        "iid_best_is_worst_rate": iid_best_is_worst / float(pairs),
        "diagnosis_accuracy": diagnosis_hits / float(cases),
        "diagnosis_by_regime": per_regime_diag,
        "mar_screen_accuracy": mar_hits / float(cases),
        "always_iid_accuracy": 1.0 / len(REGIMES),
        "driver_accuracy": driver_hits / float(cases),
        "runs": runs,
    }


def headline_numbers(evaluation):
    """Flat headline dict used by the CLI and the drift guard."""
    return {
        "diagnosis_accuracy": evaluation["diagnosis_accuracy"],
        "mar_screen_accuracy": evaluation["mar_screen_accuracy"],
        "always_iid_accuracy": evaluation["always_iid_accuracy"],
        "driver_accuracy": evaluation["driver_accuracy"],
        "gap_grouped": evaluation["gap_grouped"],
        "gap_informative": evaluation["gap_informative"],
        "gap_informative_vs_complete": (
            evaluation["gap_informative_vs_complete"]
        ),
        "reversal_rate_informative": evaluation["reversal_rate_informative"],
        "iid_best_is_worst_rate": evaluation["iid_best_is_worst_rate"],
        "gap_informative_mean_fill": evaluation["gap_informative_mean_fill"],
        "gap_informative_unit_mean": evaluation["gap_informative_unit_mean"],
        "worst_gap_informative": evaluation["worst_gap_informative"],
        "rate_iid": evaluation["missing_rate"]["iid"],
        "rate_grouped": evaluation["missing_rate"]["grouped"],
        "rate_informative": evaluation["missing_rate"]["informative"],
    }
