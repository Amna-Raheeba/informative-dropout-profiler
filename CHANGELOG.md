# Changelog

## 0.1.0 - 2026-11-22

- Initial release: seeded run-to-failure fleet simulator (six
  sensors on three physical groups) with three paired dropout
  regimes calibrated to the same missing rate - iid (MCAR),
  physically-grouped correlated outages, and failure-proximate
  informative dropout (MNAR) where each sensor's dropout probability
  rises with true degradation, scaled by sensor vulnerability.
- Profiler: reads the mask alone - co-missing lift (same-group vs
  cross-group), degradation-proxy gradient, life gradient, burstiness
  - and names the regime and its drivers (worst group; top
  health-coupled sensors). Thresholds fixed on pilot seeds 101-103,
  disjoint from the evaluation seeds.
- Downstream grid: 4 predictors (ridge, kNN, regression tree,
  single-sensor health index) x 4 imputers (mean, LOCF, linear
  interpolation, unit-mean) x 3 regimes on held-out fleets, with a
  complete-data reference.
- Headline measurements: robustness illusion gap for the iid-chosen
  configuration and per imputer cell, imputation-ranking reversal
  rate, diagnosis accuracy vs always-iid and MAR-screen baselines,
  driver identification accuracy.
- Streamlit dashboard (3 tabs), CLI demo/evaluate commands, and a
  reproduction script with a headline-drift guard.
