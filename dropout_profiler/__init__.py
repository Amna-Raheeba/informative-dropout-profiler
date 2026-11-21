"""Informative Dropout Profiler.

Audits PHM benchmarks for the missingness they actually face. Real
sensors do not drop out at random: they drop out in correlated
physical groups, and they drop out failure-proximately - the failing
machine kills its own sensors, so the data go missing exactly where
the label is most extreme (MNAR). Benchmarks that simulate only iid
dropout certify robustness the deployed model will never see.

The profiler plants three dropout regimes over identical seeded
degradation trajectories - iid (MCAR), physically-grouped correlated,
and failure-proximate informative - then measures the robustness
illusion gap for a panel of RUL predictors and imputation strategies,
the imputation-ranking reversal between regimes, and reads the mask
itself to name the mechanism and its drivers.
"""

__version__ = "0.1.0"
__author__ = "Amna Raheeba"
