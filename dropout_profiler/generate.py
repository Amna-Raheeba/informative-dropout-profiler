"""Seeded fleet simulator and the three dropout regimes.

`generate_fleet` builds complete run-to-failure trajectories. True
health falls on the curve h(t) = 1 - (t/T)^2.2; every sensor reads
baseline + gain * (1 - h) + noise, with a small per-unit gain jitter
so units are not clones.

`apply_dropout` then hides part of the panel under one of three
regimes, calibrated to the same overall missing rate (~25%):

- ``iid`` (MCAR): every cell drops independently. This is the regime
  almost every PHM robustness benchmark actually simulates.
- ``grouped`` (correlated): each physical group suffers outage
  episodes - a start probability per idle day and a geometrically
  distributed length - during which *all* of the group's sensors are
  dark together. Dropout depends on wiring, not on the machine's
  state, so it is value-independent but strongly correlated.
- ``informative`` (MNAR, failure-proximate): each sensor's dropout
  probability rises as true health falls, scaled by the sensor's
  vulnerability - the failing machine kills its own sensors, so the
  data go missing exactly where the label is most extreme.

The regimes are *paired*: the same fleet seed produces identical
machines under every regime; only the mask changes. Regime
comparisons are therefore within-fleet.
"""

import random

from .models import (
    GROUP_MEAN_LEN,
    GROUP_OUTAGE_WEIGHT,
    GROUP_START_P,
    GROUPS,
    IID_P,
    INFO_BASE,
    INFO_SLOPE,
    REGIME_INDEX,
    SENSOR_NAMES,
    SENSOR_PARAMS,
    SENSOR_VULNERABILITY,
    MaskedUnit,
    Unit,
)

MIN_LIFETIME = 90
MAX_LIFETIME = 150
HEALTH_EXPONENT = 2.2


def generate_fleet(seed, n_units):
    """Generate a complete fleet of `n_units` run-to-failure units."""
    rng = random.Random(seed)
    fleet = []
    for unit_id in range(n_units):
        lifetime = rng.randint(MIN_LIFETIME, MAX_LIFETIME)
        health = []
        values = dict((name, []) for name in SENSOR_NAMES)
        gain_jitter = dict(
            (name, rng.uniform(0.9, 1.1)) for name in SENSOR_NAMES
        )
        for day in range(lifetime):
            frac = day / float(lifetime)
            h = 1.0 - frac ** HEALTH_EXPONENT
            health.append(h)
            for name in SENSOR_NAMES:
                base, gain, noise = SENSOR_PARAMS[name]
                level = base + gain * gain_jitter[name] * (1.0 - h)
                values[name].append(level + rng.gauss(0.0, noise))
        fleet.append(Unit(unit_id, lifetime, health, values))
    return fleet


def _dropout_rng(seed, regime):
    # A stream per (fleet seed, regime): paired fleets share machines
    # but each regime draws its own mask randomness.
    return random.Random(seed * 7919 + REGIME_INDEX[regime] * 104729 + 13)


def _mask_iid(unit, rng):
    mask = {}
    for name in SENSOR_NAMES:
        mask[name] = [rng.random() < IID_P for _ in range(unit.lifetime)]
    return mask


def _mask_grouped(unit, rng):
    mask = dict((name, [False] * unit.lifetime) for name in SENSOR_NAMES)
    for group, sensors in GROUPS.items():
        start_p = GROUP_START_P * GROUP_OUTAGE_WEIGHT[group]
        day = 0
        while day < unit.lifetime:
            if rng.random() < start_p:
                # Geometric outage length with mean GROUP_MEAN_LEN.
                length = 1
                while rng.random() > 1.0 / GROUP_MEAN_LEN:
                    length += 1
                for d in range(day, min(unit.lifetime, day + length)):
                    for name in sensors:
                        mask[name][d] = True
                day += length
            else:
                day += 1
    return mask


def _mask_informative(unit, rng):
    mask = {}
    for name in SENSOR_NAMES:
        vuln = SENSOR_VULNERABILITY[name]
        col = []
        for day in range(unit.lifetime):
            degradation = 1.0 - unit.health[day]
            p = INFO_BASE + INFO_SLOPE * vuln * degradation ** 3
            if p > 0.92:
                p = 0.92
            col.append(rng.random() < p)
        mask[name] = col
    return mask


_MASKERS = {
    "iid": _mask_iid,
    "grouped": _mask_grouped,
    "informative": _mask_informative,
}


def apply_dropout(fleet, regime, seed):
    """Apply one dropout regime to a complete fleet.

    Returns (masked_units, truth) where truth records the planted
    driver, the target group/sensor rankings, and the achieved
    missing rates. The truth is scoring material only.
    """
    rng = _dropout_rng(seed, regime)
    masker = _MASKERS[regime]
    masked_units = []
    n_cells = 0
    n_missing = 0
    per_sensor_missing = dict((name, 0) for name in SENSOR_NAMES)
    for unit in fleet:
        mask = masker(unit, rng)
        masked = {}
        for name in SENSOR_NAMES:
            col = []
            for day in range(unit.lifetime):
                if mask[name][day]:
                    col.append(None)
                    per_sensor_missing[name] += 1
                    n_missing += 1
                else:
                    col.append(unit.values[name][day])
                n_cells += 1
            masked[name] = col
        masked_units.append(MaskedUnit(unit, masked, mask))
    per_sensor_rate = dict(
        (
            name,
            per_sensor_missing[name]
            / float(sum(u.lifetime for u in fleet)),
        )
        for name in SENSOR_NAMES
    )
    truth = {
        "regime": regime,
        "driver": {"iid": "none", "grouped": "group",
                   "informative": "health"}[regime],
        "worst_group": "bus_b",
        "top_vulnerable": ("vibration", "temperature"),
        "achieved_rate": n_missing / float(n_cells),
        "per_sensor_rate": per_sensor_rate,
    }
    return masked_units, truth
