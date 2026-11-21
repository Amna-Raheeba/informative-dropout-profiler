"""Example: profile one fleet under each regime and print verdicts.

Fast path - no predictor grid. Shows the profiler naming the planted
mechanism and its drivers from the mask alone.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dropout_profiler.generate import apply_dropout, generate_fleet
from dropout_profiler.models import REGIMES
from dropout_profiler.profile import profile_fleet


def main():
    fleet = generate_fleet(7, 36)
    for regime in REGIMES:
        masked, truth = apply_dropout(fleet, regime, 7)
        profile = profile_fleet(masked)
        print(
            "planted %-11s -> diagnosis %-11s (driver %s) | "
            "rate %.3f | same-lift %.2f cross-lift %.2f | "
            "proxy gradient %+.3f | worst group %s | "
            "top gradient sensors %s"
            % (
                regime,
                profile["diagnosis"],
                profile["driver"],
                profile["missing_rate"],
                profile["same_group_lift"],
                profile["cross_group_lift"],
                profile["proxy_gradient"],
                profile["worst_group"],
                ", ".join(profile["top_gradient_sensors"]),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
