import pytest

from dropout_profiler.features import (
    build_rows,
    feature_names,
    impute_series,
    sensor_means,
)
from dropout_profiler.models import MaskedUnit, SENSOR_NAMES, Unit


def test_impute_mean_exact():
    series = [1.0, None, 3.0, None]
    assert impute_series(series, "mean", 10.0) == [1.0, 10.0, 3.0, 10.0]


def test_impute_unit_mean_exact():
    series = [1.0, None, 3.0, None]
    assert impute_series(series, "unit_mean", 10.0) == [1.0, 2.0, 3.0, 2.0]


def test_impute_locf_exact_with_leading_gap():
    series = [None, 2.0, None, None, 5.0, None]
    assert impute_series(series, "locf", 9.0) == [2.0, 2.0, 2.0, 2.0, 5.0, 5.0]


def test_impute_interp_exact_midpoints_and_edges():
    series = [None, 0.0, None, None, 6.0, None]
    assert impute_series(series, "interp", 9.0) == [0.0, 0.0, 2.0, 4.0, 6.0, 6.0]


def test_impute_all_missing_falls_back_to_fleet_mean():
    series = [None, None, None]
    for method in ("mean", "locf", "interp", "unit_mean"):
        assert impute_series(series, method, 4.5) == [4.5, 4.5, 4.5]


def test_impute_unknown_method_raises():
    with pytest.raises(ValueError):
        impute_series([1.0, None], "spline", 0.0)


def test_impute_leaves_observed_values_untouched():
    series = [1.5, None, -2.5, None, 0.5]
    for method in ("mean", "locf", "interp", "unit_mean"):
        out = impute_series(series, method, 0.0)
        assert out[0] == 1.5 and out[2] == -2.5 and out[4] == 0.5
        assert all(v is not None for v in out)


def _toy_masked_unit(lifetime=40):
    values = dict(
        (name, [float(i) for i in range(lifetime)]) for name in SENSOR_NAMES
    )
    unit = Unit(0, lifetime, [1.0] * lifetime, values)
    masked = dict((name, list(values[name])) for name in SENSOR_NAMES)
    mask = dict((name, [False] * lifetime) for name in SENSOR_NAMES)
    return MaskedUnit(unit, masked, mask)


def test_sensor_means_toy():
    mu = _toy_masked_unit(10)
    means = sensor_means([mu])
    for name in SENSOR_NAMES:
        assert means[name] == 4.5


def test_build_rows_count_and_target():
    mu = _toy_masked_unit(40)
    means = sensor_means([mu])
    X, y = build_rows([mu], "mean", means, stride=5)
    expected_days = list(range(10, 40 - 3 + 1, 5))
    assert len(X) == len(expected_days)
    assert len(y) == len(expected_days)
    for row, target, day in zip(X, y, expected_days):
        assert target == 40 - day
        assert row[-1] == float(day)


def test_build_rows_feature_shape():
    mu = _toy_masked_unit(40)
    means = sensor_means([mu])
    X, _ = build_rows([mu], "locf", means)
    assert all(len(row) == len(feature_names()) for row in X)
    assert len(feature_names()) == 2 * len(SENSOR_NAMES) + 1


def test_feature_names_order():
    names = feature_names()
    assert names[0] == "vibration_level"
    assert names[len(SENSOR_NAMES)] == "vibration_slope"
    assert names[-1] == "age"
