"""Build the seeded demo fleet summary stored in data/cases.json.

The stored file is a compact summary (one row per unit) of the demo
fleet (seed 7) plus the regime ground-truth table and the achieved
missing rate of each regime on that fleet. Full trajectories are
always regenerated deterministically from the seed, never stored.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dropout_profiler.generate import apply_dropout, generate_fleet
from dropout_profiler.models import (
    GROUPS,
    REGIME_DRIVER,
    REGIMES,
    SENSOR_VULNERABILITY,
)


def main():
    fleet = generate_fleet(7, 36)
    regimes = {}
    for regime in REGIMES:
        _, truth = apply_dropout(fleet, regime, 7)
        regimes[regime] = {
            "driver": REGIME_DRIVER[regime],
            "achieved_rate": round(truth["achieved_rate"], 4),
            "per_sensor_rate": dict(
                (k, round(v, 4)) for k, v in truth["per_sensor_rate"].items()
            ),
        }
    payload = {
        "seed": 7,
        "n_units": len(fleet),
        "groups": dict((g, list(s)) for g, s in GROUPS.items()),
        "sensor_vulnerability": SENSOR_VULNERABILITY,
        "regimes": regimes,
        "units": [unit.to_dict() for unit in fleet],
    }
    out = os.path.join(os.path.dirname(__file__), "..", "data", "cases.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    print("Wrote %s (%d units)" % (out, len(fleet)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
