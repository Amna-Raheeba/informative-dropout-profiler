from dropout_profiler.evaluate import (
    headline_numbers,
    run_evaluation,
    run_single_seed,
)
from dropout_profiler.models import IMPUTERS, PREDICTORS, REGIMES


def test_single_seed_structure():
    result = run_single_seed(301, n_train=10, n_test=8)
    assert result["seed"] == 301
    for predictor in PREDICTORS:
        assert result["complete_rmse"][predictor] > 0
    for regime in REGIMES:
        block = result["regimes"][regime]
        assert block["truth"]["regime"] == regime
        assert 0.0 < block["truth"]["achieved_rate"] < 0.6
        for imputer in IMPUTERS:
            for predictor in PREDICTORS:
                assert block["rmse"][imputer][predictor] > 0


def test_tiny_evaluation_aggregates():
    evaluation = run_evaluation(seeds=(301, 302), n_train=10, n_test=8)
    assert evaluation["n_runs"] == 2
    for regime in REGIMES:
        assert 0.0 < evaluation["missing_rate"][regime] < 0.6
        for imputer in IMPUTERS:
            for predictor in PREDICTORS:
                assert evaluation["rmse_mean"][regime][imputer][predictor] > 0
    assert 0.0 <= evaluation["diagnosis_accuracy"] <= 1.0
    assert 0.0 <= evaluation["reversal_rate_informative"] <= 1.0
    assert evaluation["always_iid_accuracy"] == 1.0 / 3.0


def test_headline_numbers_keys():
    evaluation = run_evaluation(seeds=(301,), n_train=10, n_test=8)
    headline = headline_numbers(evaluation)
    for key in (
        "diagnosis_accuracy",
        "mar_screen_accuracy",
        "always_iid_accuracy",
        "driver_accuracy",
        "gap_grouped",
        "gap_informative",
        "gap_informative_mean_fill",
        "gap_informative_unit_mean",
        "worst_gap_informative",
        "reversal_rate_informative",
        "rate_iid",
        "rate_grouped",
        "rate_informative",
    ):
        assert key in headline, key


def test_evaluation_is_deterministic():
    a = run_evaluation(seeds=(301,), n_train=10, n_test=8)
    b = run_evaluation(seeds=(301,), n_train=10, n_test=8)
    assert headline_numbers(a) == headline_numbers(b)
