"""Imputation and landmark-row feature extraction.

The downstream task is remaining-useful-life (RUL) regression at
landmark days: for each unit and each sampled day, predict days until
failure from the sensor panel *as observed* - that is, after dropout
and after an imputation strategy has filled the holes. Imputation is
where the missingness mechanism bites: a strategy that is harmless
under iid dropout can fabricate healthy-looking values exactly where
an informative regime has hidden the sickest readings.

Four imputers, all per unit and per sensor over the time axis:

- ``mean``: fill with the sensor's fleet-wide observed mean
  (computed on the training fleet).
- ``locf``: last observation carried forward; the leading gap is
  filled from the first observation (next-observation-carried-back),
  falling back to the fleet mean for a never-observed sensor.
- ``interp``: linear interpolation between bracketing observations;
  edges take the nearest observation; never-observed falls back to
  the fleet mean.
- ``unit_mean``: fill with the unit's own observed mean for that
  sensor; never-observed falls back to the fleet mean.

Features per landmark row: each sensor's imputed level, each sensor's
10-day slope of the imputed series, and the unit's age in days.
Target: RUL = lifetime - day.
"""

from .models import SENSOR_NAMES

STRIDE = 5
MIN_AGE = 10
MIN_RUL = 3
SLOPE_WINDOW = 10


def feature_names():
    names = ["%s_level" % name for name in SENSOR_NAMES]
    names += ["%s_slope" % name for name in SENSOR_NAMES]
    names.append("age")
    return names


def sensor_means(masked_units):
    """Fleet-wide observed mean per sensor (training statistics)."""
    totals = dict((name, 0.0) for name in SENSOR_NAMES)
    counts = dict((name, 0) for name in SENSOR_NAMES)
    for mu in masked_units:
        for name in SENSOR_NAMES:
            for value in mu.masked[name]:
                if value is not None:
                    totals[name] += value
                    counts[name] += 1
    return dict(
        (name, totals[name] / counts[name] if counts[name] else 0.0)
        for name in SENSOR_NAMES
    )


def impute_series(series, method, fleet_mean):
    """Fill the None entries of one sensor series for one unit."""
    observed = [v for v in series if v is not None]
    if not observed:
        return [fleet_mean] * len(series)
    if method == "mean":
        return [fleet_mean if v is None else v for v in series]
    if method == "unit_mean":
        own = sum(observed) / len(observed)
        return [own if v is None else v for v in series]
    if method == "locf":
        out = []
        last = observed[0]
        for v in series:
            if v is None:
                out.append(last)
            else:
                out.append(v)
                last = v
        return out
    if method == "interp":
        out = list(series)
        n = len(series)
        idx = [i for i, v in enumerate(series) if v is not None]
        for k, i in enumerate(idx):
            if k == 0:
                for j in range(i):
                    out[j] = series[i]
            else:
                prev = idx[k - 1]
                gap = i - prev
                for j in range(prev + 1, i):
                    frac = (j - prev) / float(gap)
                    out[j] = series[prev] + frac * (series[i] - series[prev])
            if k == len(idx) - 1:
                for j in range(i + 1, n):
                    out[j] = series[i]
        return out
    raise ValueError("unknown imputer: %s" % method)


def impute_unit(masked_unit, method, means):
    """Impute every sensor series of one masked unit."""
    return dict(
        (name, impute_series(masked_unit.masked[name], method, means[name]))
        for name in SENSOR_NAMES
    )


def build_rows(masked_units, method, means, stride=STRIDE):
    """Build landmark rows (X, y) from masked units under an imputer.

    `means` must come from the training fleet so test rows never see
    test statistics.
    """
    X = []
    y = []
    for mu in masked_units:
        filled = impute_unit(mu, method, means)
        for day in range(MIN_AGE, mu.lifetime - MIN_RUL + 1, stride):
            row = []
            for name in SENSOR_NAMES:
                row.append(filled[name][day])
            for name in SENSOR_NAMES:
                back = max(0, day - SLOPE_WINDOW)
                span = day - back
                if span == 0:
                    row.append(0.0)
                else:
                    row.append(
                        (filled[name][day] - filled[name][back]) / float(span)
                    )
            row.append(float(day))
            X.append(row)
            y.append(float(mu.lifetime - day))
    return X, y
