"""The profiler: read the mask, name the mechanism and its drivers.

Everything here uses only the masked panel (plus, for the
failure-proximity statistic, each unit's lifetime - known for any
historical run-to-failure panel, which is the audit setting). The
profiler never sees true health or the planted parameters; the
evaluation scores its verdicts against that ground truth.

Three statistics separate the three regimes:

- **Co-missing lift.** For a pair of sensors, lift = P(both missing)
  / (P(a missing) * P(b missing)). Under iid dropout lift is ~1 for
  every pair. Under grouped dropout, pairs on the same bus have lift
  far above pairs across buses. Under informative dropout *all* pairs
  lift together (shared health driver), so the same-group/cross-group
  contrast - not the level - is the grouped signature.
- **Degradation-proxy gradient.** Standardise each observed reading
  by the sensor's fleet mean/sd and average the observed sensors per
  unit-day: a degradation proxy readable from the data itself. The
  gradient is the missing rate in the sickest proxy quartile minus
  the healthiest. Only informative dropout produces a large positive
  gradient, because only there does missingness track the machine's
  state.
- **Burstiness.** Mean missing-run length. Grouped outages and
  failure-proximate death both burst; iid dropout does not. Used as
  corroboration, not as the decision rule.

Diagnosis thresholds were fixed on pilot fleets (seeds 101-103),
disjoint from the evaluation seeds, and frozen in this module.
"""

import itertools
import math

from .features import sensor_means
from .models import GROUPS, SENSOR_GROUP, SENSOR_NAMES

GROUPED_LIFT_MIN = 2.5
GROUPED_LIFT_RATIO = 1.8
INFORMATIVE_GRADIENT_MIN = 0.12


def missing_rates(masked_units):
    """Overall, per-sensor and per-group missing rates."""
    total = 0
    missing = 0
    per_sensor = dict((name, [0, 0]) for name in SENSOR_NAMES)
    for mu in masked_units:
        for name in SENSOR_NAMES:
            for flag in mu.mask[name]:
                per_sensor[name][1] += 1
                total += 1
                if flag:
                    per_sensor[name][0] += 1
                    missing += 1
    sensor_rate = dict(
        (name, per_sensor[name][0] / float(per_sensor[name][1]))
        for name in SENSOR_NAMES
    )
    group_rate = {}
    for group, sensors in GROUPS.items():
        num = sum(per_sensor[s][0] for s in sensors)
        den = sum(per_sensor[s][1] for s in sensors)
        group_rate[group] = num / float(den)
    return {
        "overall": missing / float(total),
        "per_sensor": sensor_rate,
        "per_group": group_rate,
    }


def mean_run_length(masked_units):
    """Mean length of consecutive-missing runs, pooled over sensors."""
    runs = []
    for mu in masked_units:
        for name in SENSOR_NAMES:
            current = 0
            for flag in mu.mask[name]:
                if flag:
                    current += 1
                elif current:
                    runs.append(current)
                    current = 0
            if current:
                runs.append(current)
    if not runs:
        return 0.0
    return sum(runs) / float(len(runs))


def _pair_lift(masked_units, a, b):
    both = 0
    pa = 0
    pb = 0
    n = 0
    for mu in masked_units:
        for day in range(mu.lifetime):
            fa = mu.mask[a][day]
            fb = mu.mask[b][day]
            n += 1
            pa += fa
            pb += fb
            both += fa and fb
    if pa == 0 or pb == 0 or n == 0:
        return 1.0
    return (both / float(n)) / ((pa / float(n)) * (pb / float(n)))


def co_missing_lift(masked_units):
    """(same-group lift, cross-group lift), averaged over pairs."""
    same = []
    cross = []
    for a, b in itertools.combinations(SENSOR_NAMES, 2):
        lift = _pair_lift(masked_units, a, b)
        if SENSOR_GROUP[a] == SENSOR_GROUP[b]:
            same.append(lift)
        else:
            cross.append(lift)
    return (sum(same) / len(same), sum(cross) / len(cross))


def _sensor_stds(masked_units, means):
    stds = {}
    for name in SENSOR_NAMES:
        vals = [
            v for mu in masked_units for v in mu.masked[name]
            if v is not None
        ]
        if not vals:
            stds[name] = 1.0
            continue
        var = sum((v - means[name]) ** 2 for v in vals) / len(vals)
        stds[name] = math.sqrt(var) if var > 0 else 1.0
    return stds


def proxy_series(masked_units):
    """Per unit-day degradation proxy from observed readings only.

    Returns a list of (proxy, missing_fraction) per unit-day with at
    least one observed sensor. Higher proxy = sicker machine.
    """
    means = sensor_means(masked_units)
    stds = _sensor_stds(masked_units, means)
    rows = []
    for mu in masked_units:
        for day in range(mu.lifetime):
            observed = []
            n_missing = 0
            for name in SENSOR_NAMES:
                value = mu.masked[name][day]
                if value is None:
                    n_missing += 1
                else:
                    observed.append((value - means[name]) / stds[name])
            if observed:
                rows.append(
                    (sum(observed) / len(observed),
                     n_missing / float(len(SENSOR_NAMES)))
                )
    return rows


def _quartile_gradient(pairs):
    """Missing rate in the top proxy quartile minus the bottom."""
    if len(pairs) < 8:
        return 0.0
    ordered = sorted(pairs)
    q = len(ordered) // 4
    low = ordered[:q]
    high = ordered[-q:]
    low_rate = sum(p[1] for p in low) / len(low)
    high_rate = sum(p[1] for p in high) / len(high)
    return high_rate - low_rate


def proxy_gradient(masked_units):
    return _quartile_gradient(proxy_series(masked_units))


def sensor_proxy_gradients(masked_units):
    """Per-sensor missingness gradient over the panel proxy.

    For each sensor: its missing rate on the sickest-proxy-quartile
    unit-days minus its rate on the healthiest-quartile unit-days.
    Under informative dropout this ranks sensors by vulnerability.
    """
    means = sensor_means(masked_units)
    stds = _sensor_stds(masked_units, means)
    per_day = []
    for mu in masked_units:
        for day in range(mu.lifetime):
            observed = []
            for name in SENSOR_NAMES:
                value = mu.masked[name][day]
                if value is not None:
                    observed.append((value - means[name]) / stds[name])
            if not observed:
                continue
            proxy = sum(observed) / len(observed)
            flags = dict(
                (name, 1.0 if mu.mask[name][day] else 0.0)
                for name in SENSOR_NAMES
            )
            per_day.append((proxy, flags))
    if len(per_day) < 8:
        return dict((name, 0.0) for name in SENSOR_NAMES)
    per_day.sort(key=lambda row: row[0])
    q = len(per_day) // 4
    low = per_day[:q]
    high = per_day[-q:]
    gradients = {}
    for name in SENSOR_NAMES:
        lo = sum(row[1][name] for row in low) / len(low)
        hi = sum(row[1][name] for row in high) / len(high)
        gradients[name] = hi - lo
    return gradients


def life_gradient(masked_units):
    """Missing rate in the last life-quartile minus the first.

    Uses lifetimes, which a run-to-failure audit panel knows.
    Corroborates the proxy gradient for informative dropout.
    """
    first = [0, 0]
    last = [0, 0]
    for mu in masked_units:
        for name in SENSOR_NAMES:
            for day in range(mu.lifetime):
                frac = day / float(mu.lifetime)
                if frac < 0.25:
                    first[1] += 1
                    first[0] += mu.mask[name][day]
                elif frac >= 0.75:
                    last[1] += 1
                    last[0] += mu.mask[name][day]
    if not first[1] or not last[1]:
        return 0.0
    return last[0] / float(last[1]) - first[0] / float(first[1])


def profile_fleet(masked_units):
    """Full profile of one masked fleet: statistics + diagnosis."""
    rates = missing_rates(masked_units)
    same_lift, cross_lift = co_missing_lift(masked_units)
    grad = proxy_gradient(masked_units)
    stats = {
        "missing_rate": rates["overall"],
        "per_sensor_rate": rates["per_sensor"],
        "per_group_rate": rates["per_group"],
        "same_group_lift": same_lift,
        "cross_group_lift": cross_lift,
        "proxy_gradient": grad,
        "life_gradient": life_gradient(masked_units),
        "mean_run_length": mean_run_length(masked_units),
        "sensor_gradients": sensor_proxy_gradients(masked_units),
    }
    stats["diagnosis"] = diagnose(stats)
    stats["driver"] = {
        "iid": "none",
        "grouped": "group",
        "informative": "health",
    }[stats["diagnosis"]]
    stats["worst_group"] = max(
        rates["per_group"], key=lambda g: rates["per_group"][g]
    )
    ranked = sorted(
        stats["sensor_gradients"],
        key=lambda s: stats["sensor_gradients"][s],
        reverse=True,
    )
    stats["top_gradient_sensors"] = tuple(ranked[:2])
    return stats


def diagnose(stats):
    """Name the dropout regime from profile statistics."""
    same = stats["same_group_lift"]
    cross = stats["cross_group_lift"]
    if same >= GROUPED_LIFT_MIN and same >= GROUPED_LIFT_RATIO * cross:
        return "grouped"
    if stats["proxy_gradient"] >= INFORMATIVE_GRADIENT_MIN:
        return "informative"
    return "iid"


def diagnose_mar_screen(stats):
    """Baseline diagnosis: an observed-covariate (MAR) screen only.

    Flags informative dropout when missingness tracks the observed
    degradation proxy, otherwise calls everything iid. It has no
    grouped category at all - correlated wiring failures are
    invisible to it. This is the strongest quick check a practitioner
    typically runs before assuming MCAR.
    """
    if stats["proxy_gradient"] >= INFORMATIVE_GRADIENT_MIN:
        return "informative"
    return "iid"
