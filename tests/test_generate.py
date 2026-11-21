from dropout_profiler.generate import (
    MAX_LIFETIME,
    MIN_LIFETIME,
    apply_dropout,
    generate_fleet,
)
from dropout_profiler.models import REGIMES, SENSOR_NAMES


def test_generation_is_deterministic():
    a = generate_fleet(11, 6)
    b = generate_fleet(11, 6)
    for ua, ub in zip(a, b):
        assert ua.lifetime == ub.lifetime
        assert ua.values["vibration"] == ub.values["vibration"]


def test_different_seeds_differ():
    a = generate_fleet(11, 6)
    b = generate_fleet(12, 6)
    assert a[0].values["vibration"] != b[0].values["vibration"]


def test_fleet_shape_and_lifetimes():
    fleet = generate_fleet(3, 15)
    assert len(fleet) == 15
    for unit in fleet:
        assert MIN_LIFETIME <= unit.lifetime <= MAX_LIFETIME
        assert len(unit.health) == unit.lifetime
        for name in SENSOR_NAMES:
            assert len(unit.values[name]) == unit.lifetime


def test_health_declines_from_one_to_zero():
    fleet = generate_fleet(3, 10)
    for unit in fleet:
        assert unit.health[0] > 0.99
        assert unit.health[-1] < 0.05
        assert all(
            unit.health[i] >= unit.health[i + 1]
            for i in range(unit.lifetime - 1)
        )


def test_regimes_are_paired_on_the_same_machines():
    fleet = generate_fleet(9, 8)
    masked_iid, _ = apply_dropout(fleet, "iid", 9)
    masked_inf, _ = apply_dropout(fleet, "informative", 9)
    for a, b in zip(masked_iid, masked_inf):
        assert a.unit.values["pressure"] == b.unit.values["pressure"]


def test_dropout_is_deterministic():
    fleet = generate_fleet(9, 8)
    m1, t1 = apply_dropout(fleet, "grouped", 9)
    m2, t2 = apply_dropout(fleet, "grouped", 9)
    assert t1["achieved_rate"] == t2["achieved_rate"]
    assert m1[0].mask == m2[0].mask


def test_masked_values_match_complete_where_observed():
    fleet = generate_fleet(9, 8)
    masked, _ = apply_dropout(fleet, "iid", 9)
    for mu in masked:
        for name in SENSOR_NAMES:
            for day in range(mu.lifetime):
                if mu.mask[name][day]:
                    assert mu.masked[name][day] is None
                else:
                    assert mu.masked[name][day] == mu.unit.values[name][day]


def test_achieved_rates_are_matched_across_regimes(fleet):
    rates = {}
    for regime in REGIMES:
        _, truth = apply_dropout(fleet, regime, 5)
        rates[regime] = truth["achieved_rate"]
    for regime in REGIMES:
        assert 0.15 < rates[regime] < 0.33, (regime, rates[regime])
    assert max(rates.values()) - min(rates.values()) < 0.08


def test_grouped_sensors_drop_together(masked_by_regime):
    masked, _ = masked_by_regime["grouped"]
    both = 0
    either = 0
    for mu in masked:
        for day in range(mu.lifetime):
            a = mu.mask["vibration"][day]
            b = mu.mask["temperature"][day]
            if a or b:
                either += 1
            if a and b:
                both += 1
    # Same-bus sensors are dark together or not at all.
    assert both == either


def test_informative_dropout_is_failure_proximate(masked_by_regime):
    masked, _ = masked_by_regime["informative"]
    first = [0, 0]
    last = [0, 0]
    for mu in masked:
        for name in SENSOR_NAMES:
            for day in range(mu.lifetime):
                frac = day / float(mu.lifetime)
                if frac < 0.2:
                    first[1] += 1
                    first[0] += mu.mask[name][day]
                elif frac >= 0.8:
                    last[1] += 1
                    last[0] += mu.mask[name][day]
    assert last[0] / last[1] > 3.0 * (first[0] / first[1])


def test_informative_dropout_ranks_sensors_by_vulnerability(masked_by_regime):
    _, truth = masked_by_regime["informative"]
    rates = truth["per_sensor_rate"]
    assert rates["vibration"] > rates["humidity"]
    assert rates["temperature"] > rates["acoustic"]


def test_grouped_worst_group_is_bus_b(masked_by_regime):
    _, truth = masked_by_regime["grouped"]
    rates = truth["per_sensor_rate"]
    bus_b = (rates["pressure"] + rates["current"]) / 2.0
    bus_a = (rates["vibration"] + rates["temperature"]) / 2.0
    zone_c = (rates["acoustic"] + rates["humidity"]) / 2.0
    assert bus_b > bus_a
    assert bus_b > zone_c
