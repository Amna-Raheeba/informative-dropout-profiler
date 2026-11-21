from dropout_profiler.models import MaskedUnit, SENSOR_NAMES, Unit
from dropout_profiler.profile import (
    co_missing_lift,
    diagnose,
    diagnose_mar_screen,
    life_gradient,
    mean_run_length,
    missing_rates,
    profile_fleet,
    proxy_gradient,
)


def _handmade_unit(missing_days):
    lifetime = 20
    values = dict(
        (name, [0.0] * lifetime) for name in SENSOR_NAMES
    )
    unit = Unit(0, lifetime, [1.0] * lifetime, values)
    mask = dict(
        (name, [day in missing_days for day in range(lifetime)])
        for name in SENSOR_NAMES
    )
    masked = dict(
        (
            name,
            [None if mask[name][d] else 0.0 for d in range(lifetime)],
        )
        for name in SENSOR_NAMES
    )
    return MaskedUnit(unit, masked, mask)


def test_missing_rates_handmade():
    mu = _handmade_unit({0, 1, 2, 3})
    rates = missing_rates([mu])
    assert rates["overall"] == 0.2
    assert all(v == 0.2 for v in rates["per_sensor"].values())
    assert all(v == 0.2 for v in rates["per_group"].values())


def test_mean_run_length_handmade():
    mu = _handmade_unit({2, 3, 4, 9})
    # Every sensor has runs of 3 and 1 -> mean 2.0.
    assert mean_run_length([mu]) == 2.0


def test_diagnose_grouped_from_stats():
    stats = {
        "same_group_lift": 4.0,
        "cross_group_lift": 1.0,
        "proxy_gradient": 0.01,
    }
    assert diagnose(stats) == "grouped"


def test_diagnose_informative_from_stats():
    stats = {
        "same_group_lift": 2.2,
        "cross_group_lift": 2.1,
        "proxy_gradient": 0.30,
    }
    assert diagnose(stats) == "informative"


def test_diagnose_iid_from_stats():
    stats = {
        "same_group_lift": 1.05,
        "cross_group_lift": 0.98,
        "proxy_gradient": -0.01,
    }
    assert diagnose(stats) == "iid"


def test_diagnose_grouped_needs_contrast_not_level():
    # All pairs lift together (shared driver) -> not grouped.
    stats = {
        "same_group_lift": 3.0,
        "cross_group_lift": 2.9,
        "proxy_gradient": 0.0,
    }
    assert diagnose(stats) == "iid"


def test_mar_screen_has_no_grouped_category():
    stats = {
        "same_group_lift": 9.0,
        "cross_group_lift": 1.0,
        "proxy_gradient": 0.0,
    }
    assert diagnose_mar_screen(stats) == "iid"
    stats["proxy_gradient"] = 0.25
    assert diagnose_mar_screen(stats) == "informative"


def test_profile_diagnoses_each_regime(masked_by_regime):
    for regime in ("iid", "grouped", "informative"):
        masked, _ = masked_by_regime[regime]
        profile = profile_fleet(masked)
        assert profile["diagnosis"] == regime, (regime, profile)


def test_profile_grouped_signature(masked_by_regime):
    masked, _ = masked_by_regime["grouped"]
    same, cross = co_missing_lift(masked)
    assert same > 2.5
    assert same > 1.8 * cross
    profile = profile_fleet(masked)
    assert profile["worst_group"] == "bus_b"
    assert profile["mean_run_length"] > 3.0


def test_profile_informative_signature(masked_by_regime):
    masked, _ = masked_by_regime["informative"]
    assert proxy_gradient(masked) > 0.12
    assert life_gradient(masked) > 0.3
    profile = profile_fleet(masked)
    assert set(profile["top_gradient_sensors"]) == {
        "vibration",
        "temperature",
    }


def test_profile_iid_is_quiet(masked_by_regime):
    masked, _ = masked_by_regime["iid"]
    same, cross = co_missing_lift(masked)
    assert 0.8 < same < 1.3
    assert abs(proxy_gradient(masked)) < 0.06
    assert abs(life_gradient(masked)) < 0.06
