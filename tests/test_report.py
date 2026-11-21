from dropout_profiler.evaluate import (
    headline_numbers,
    run_evaluation,
    run_single_seed,
)
from dropout_profiler.report import (
    render_evaluation_summary,
    render_single_seed_report,
)


def test_single_seed_report_mentions_all_regimes():
    result = run_single_seed(301, n_train=10, n_test=8)
    text = render_single_seed_report(result)
    assert isinstance(text, str)
    for regime in ("iid", "grouped", "informative"):
        assert regime in text
    assert "diagnosis" in text
    assert "Complete-data reference" in text


def test_evaluation_summary_mentions_headline_objects():
    evaluation = run_evaluation(seeds=(301,), n_train=10, n_test=8)
    text = render_evaluation_summary(
        evaluation, headline_numbers(evaluation)
    )
    assert "Robustness illusion gap" in text
    assert "Imputation-ranking reversal" in text
    assert "Diagnosis and drivers" in text
    assert "always-iid" in text
