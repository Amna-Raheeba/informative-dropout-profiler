import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "evaluation.json")


def test_stored_results_exist_and_parse():
    assert os.path.isfile(RESULTS)
    with open(RESULTS) as fh:
        payload = json.load(fh)
    for key in ("seeds", "n_runs", "missing_rate", "rmse_mean",
                "complete_rmse_mean", "diagnosis_by_regime", "headline"):
        assert key in payload, key


def test_stored_headline_keys():
    with open(RESULTS) as fh:
        payload = json.load(fh)
    headline = payload["headline"]
    for key in (
        "diagnosis_accuracy",
        "mar_screen_accuracy",
        "always_iid_accuracy",
        "driver_accuracy",
        "gap_grouped",
        "gap_informative",
        "gap_informative_mean_fill",
        "worst_gap_informative",
        "reversal_rate_informative",
        "rate_iid",
        "rate_grouped",
        "rate_informative",
    ):
        assert key in headline, key


def test_stored_headline_values_sane():
    with open(RESULTS) as fh:
        payload = json.load(fh)
    headline = payload["headline"]
    # The profiler names the planted regime far better than the
    # baselines it is compared against.
    assert headline["diagnosis_accuracy"] >= 0.9
    assert headline["mar_screen_accuracy"] < headline["diagnosis_accuracy"]
    assert headline["always_iid_accuracy"] < 0.34
    # Rates were calibrated to match across regimes.
    rates = [
        headline["rate_iid"],
        headline["rate_grouped"],
        headline["rate_informative"],
    ]
    assert max(rates) - min(rates) < 0.08
    # The illusion is real and concentrates in the naive fill.
    assert headline["gap_informative"] > 0.0
    assert headline["gap_informative_mean_fill"] > headline["gap_informative"]
    assert headline["worst_gap_informative"] > 3.0
    assert 0.0 < headline["reversal_rate_informative"] < 1.0
