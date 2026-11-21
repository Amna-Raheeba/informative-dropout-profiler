# Evaluation summary

6 paired seeds (201, 202, 203, 204, 205, 206); train fleets of 36 units, test fleets of 20 units. Every regime masks the same machines - only the missingness mechanism changes.

## Achieved missing rates (calibrated to match)

- iid: 0.249
- grouped: 0.250
- informative: 0.232

## Robustness illusion gap

Configuration chosen under iid dropout (best imputer per predictor), re-scored under the other regimes. Gap = RMSE under the regime minus RMSE under iid, in days.

- grouped gap: +0.44 days
- informative gap: +0.49 days
- informative vs complete data: +0.08 days

Per-cell gaps (regime RMSE minus iid RMSE, days) show where the illusion lives: it concentrates in the naive fills. Under informative dropout the mean-fill gap averages +1.93 days across predictors and the unit-mean gap +1.95, while the worst cell (mean/single_sensor) degrades by +6.41 days. Under grouped dropout the same fills barely move - the damage is mechanism-specific, and an iid benchmark cannot see it.

## Imputation-ranking reversal

- best imputer changes iid -> informative in 37.5% of (seed, predictor) pairs
- best imputer changes iid -> grouped in 29.2% of pairs
- the iid-best imputer is the *worst* under informative dropout in 0.0% of pairs

## Diagnosis and drivers

- profiler diagnosis accuracy: 1.000 (iid 1.00, grouped 1.00, informative 1.00)
- MAR-screen baseline accuracy: 0.667
- always-iid baseline accuracy: 0.333
- driver identification accuracy: 1.000

## Mean RMSE by regime x imputer x predictor

### iid

| imputer | ridge | knn | tree | single_sensor |
|---|---|---|---|---|
| mean | 14.96 | 16.94 | 15.83 | 23.08 |
| locf | 14.26 | 15.62 | 14.98 | 17.67 |
| interp | 14.18 | 15.47 | 14.64 | 17.17 |
| unit_mean | 15.01 | 17.01 | 16.17 | 23.21 |

### grouped

| imputer | ridge | knn | tree | single_sensor |
|---|---|---|---|---|
| mean | 15.43 | 16.89 | 15.97 | 22.53 |
| locf | 14.69 | 16.17 | 15.40 | 18.28 |
| interp | 14.30 | 15.72 | 14.90 | 17.65 |
| unit_mean | 15.65 | 16.86 | 16.27 | 23.40 |

### informative

| imputer | ridge | knn | tree | single_sensor |
|---|---|---|---|---|
| mean | 15.98 | 16.72 | 16.31 | 29.49 |
| locf | 14.18 | 15.59 | 15.49 | 18.56 |
| interp | 14.19 | 15.42 | 15.30 | 18.32 |
| unit_mean | 16.01 | 16.82 | 16.76 | 29.60 |

### complete data (reference)

- ridge: 14.37 days
- knn: 15.95 days
- tree: 14.87 days
- single_sensor: 17.74 days
