"""Shared constants and the unit (trajectory) container.

The fleet is a panel of run-to-failure machines. Each unit is observed
daily on six sensors. The sensors sit on three physical groups (two
shared buses and one zone harness), which is what makes *grouped*
dropout physically plausible: when a bus dies, both of its sensors go
dark together.

Sensor readings rise as true health falls (vibration, temperature and
friends are degradation signals). One sensor (humidity) is a near-null
control: it barely tracks health, so a profiler that blames dropout on
it is confessing it cannot read the panel.
"""

SENSOR_NAMES = (
    "vibration",
    "temperature",
    "pressure",
    "current",
    "acoustic",
    "humidity",
)

# sensor -> (baseline level, health gain, noise sd)
# Reading = baseline + gain * (1 - health) + noise. Health is 1.0 at
# birth and 0.0 at failure, so a gain of 3.0 means the reading climbs
# three units over the unit's life.
SENSOR_PARAMS = {
    "vibration": (1.00, 3.00, 0.35),
    "temperature": (0.80, 2.20, 0.30),
    "pressure": (1.20, 1.60, 0.30),
    "current": (0.90, 1.20, 0.35),
    "acoustic": (0.60, 0.80, 0.40),
    "humidity": (0.50, 0.15, 0.50),
}

GROUPS = {
    "bus_a": ("vibration", "temperature"),
    "bus_b": ("pressure", "current"),
    "zone_c": ("acoustic", "humidity"),
}

SENSOR_GROUP = dict(
    (sensor, group)
    for group, sensors in GROUPS.items()
    for sensor in sensors
)

# Grouped-regime outage propensity per group: bus_b is the flaky bus.
# Ground truth for driver identification; the profiler must recover
# bus_b as the worst group from the masked data alone.
GROUP_OUTAGE_WEIGHT = {
    "bus_a": 0.8,
    "bus_b": 1.5,
    "zone_c": 0.7,
}

# Informative-regime vulnerability per sensor: how strongly a sensor's
# dropout probability couples to falling health. The sensors mounted
# on the failing component (vibration, temperature) die first.
# Ground truth for driver identification.
SENSOR_VULNERABILITY = {
    "vibration": 1.00,
    "temperature": 0.90,
    "pressure": 0.70,
    "current": 0.60,
    "acoustic": 0.45,
    "humidity": 0.30,
}

REGIMES = ("iid", "grouped", "informative")

# The dropout driver planted by each regime. The profiler's job is to
# name it from the masked data alone.
REGIME_DRIVER = {
    "iid": "none",
    "grouped": "group",
    "informative": "health",
}

REGIME_INDEX = dict((name, i) for i, name in enumerate(REGIMES))

IMPUTERS = ("mean", "locf", "interp", "unit_mean")

PREDICTORS = ("ridge", "knn", "tree", "single_sensor")

# Dropout regime parameters. The three regimes are calibrated to the
# same overall missing rate (~25%) so that score differences between
# regimes are mechanism effects, not volume effects. Achieved rates
# are measured and reported by the evaluation.
IID_P = 0.25

GROUP_START_P = 0.0667  # per group per idle day, before group weight
GROUP_MEAN_LEN = 5.0    # mean outage length in days

INFO_BASE = 0.07        # dropout probability at full health
INFO_SLOPE = 2.30       # added probability mass at zero health


class Unit(object):
    """One run-to-failure trajectory with complete sensor values.

    Attributes:
        unit_id: integer id within its fleet.
        lifetime: failure day T (the unit is observed on days
            0..T-1 and fails at T).
        health: list of true health values, one per day, 1.0 -> ~0.0.
        values: dict sensor -> list of readings, one per day.
    """

    def __init__(self, unit_id, lifetime, health, values):
        self.unit_id = unit_id
        self.lifetime = lifetime
        self.health = health
        self.values = values

    @property
    def n_days(self):
        return self.lifetime

    def to_dict(self):
        return {
            "unit_id": self.unit_id,
            "lifetime": self.lifetime,
            "mean_health": round(sum(self.health) / len(self.health), 4),
        }


class MaskedUnit(object):
    """A unit after a dropout regime has been applied.

    Attributes:
        unit: the underlying complete Unit (ground truth; feature
            extraction and the deployable profiler never read it -
            only the evaluation's oracle checks may).
        masked: dict sensor -> list of readings with None where the
            sensor was dark.
        mask: dict sensor -> list of bool, True where missing.
    """

    def __init__(self, unit, masked, mask):
        self.unit = unit
        self.masked = masked
        self.mask = mask

    @property
    def unit_id(self):
        return self.unit.unit_id

    @property
    def lifetime(self):
        return self.unit.lifetime

    @property
    def n_days(self):
        return self.unit.lifetime
