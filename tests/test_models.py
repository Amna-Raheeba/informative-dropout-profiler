from dropout_profiler.models import (
    GROUPS,
    GROUP_OUTAGE_WEIGHT,
    IMPUTERS,
    PREDICTORS,
    REGIME_DRIVER,
    REGIMES,
    SENSOR_GROUP,
    SENSOR_NAMES,
    SENSOR_PARAMS,
    SENSOR_VULNERABILITY,
    Unit,
)


def test_groups_partition_sensors():
    grouped = [s for sensors in GROUPS.values() for s in sensors]
    assert sorted(grouped) == sorted(SENSOR_NAMES)
    assert len(grouped) == len(set(grouped))


def test_sensor_group_lookup_complete():
    for name in SENSOR_NAMES:
        assert name in SENSOR_GROUP
        assert name in SENSOR_PARAMS
        assert name in SENSOR_VULNERABILITY


def test_vulnerability_ordering():
    assert SENSOR_VULNERABILITY["vibration"] == max(
        SENSOR_VULNERABILITY.values()
    )
    assert SENSOR_VULNERABILITY["humidity"] == min(
        SENSOR_VULNERABILITY.values()
    )
    assert (
        SENSOR_VULNERABILITY["vibration"]
        > SENSOR_VULNERABILITY["temperature"]
        > SENSOR_VULNERABILITY["humidity"]
    )


def test_bus_b_is_flakiest_group():
    assert GROUP_OUTAGE_WEIGHT["bus_b"] == max(GROUP_OUTAGE_WEIGHT.values())


def test_regime_driver_mapping():
    assert REGIME_DRIVER == {
        "iid": "none",
        "grouped": "group",
        "informative": "health",
    }
    assert set(REGIMES) == set(REGIME_DRIVER)


def test_panel_sizes():
    assert len(SENSOR_NAMES) == 6
    assert len(IMPUTERS) == 4
    assert len(PREDICTORS) == 4


def test_unit_to_dict():
    unit = Unit(3, 10, [1.0 - i / 10.0 for i in range(10)],
                dict((n, [0.0] * 10) for n in SENSOR_NAMES))
    payload = unit.to_dict()
    assert payload["unit_id"] == 3
    assert payload["lifetime"] == 10
    assert unit.n_days == 10
