# Informative Dropout Profiler

## 1. What this is

PHM models are benchmarked against the wrong missingness. The
standard robustness check - drop sensor readings uniformly at random
and re-score - simulates a world that does not exist. Real sensors
drop out **correlated** (a shared bus or harness takes its whole
group down together) and **failure-proximately**: the failing machine
kills its own sensors, so readings go missing exactly where the
label is most extreme. That second mechanism is MNAR - missing not
at random, tied to the health state the model is trying to predict -
and no iid benchmark can measure what it costs.

This profiler plants three dropout regimes over *identical* seeded
degradation trajectories, at the *same* overall missing rate, and
measures three things:

1. the **robustness illusion gap** - how much worse a benchmarked
   configuration actually scores under grouped and informative
   dropout than under the iid dropout it was certified on;
2. the **imputation-ranking reversal** - how often the imputation
   strategy that wins under iid loses under informative dropout;
3. a **diagnosis** - the profiler reads the mask alone and names
   which mechanism produced it, and which variables drive it.

## 2. The novel mechanism

1. **Paired regimes, matched rates.** One fleet of run-to-failure
   machines is generated once per seed, then masked three ways:
   iid (MCAR), physically-grouped outages, and failure-proximate
   informative dropout whose per-sensor probability rises with true
   degradation, scaled by a per-sensor vulnerability. Achieved rates
   land within 0.018 of each other (Section 6), so every score
   difference is a mechanism effect, not a volume effect.
2. **A full downstream grid.** 4 predictors x 4 imputers x 3 regimes,
   trained on masked fleets and scored on separately seeded masked
   test fleets, plus a complete-data reference. The illusion gap is
   reported per cell and for the configuration an iid benchmark
   would actually choose.
3. **Mask forensics.** Three statistics separate the regimes from
   the observed panel alone: *co-missing lift* (same-group pairs vs
   cross-group pairs - grouped dropout lifts only the former,
   informative dropout lifts all pairs together), the
   *degradation-proxy gradient* (missing rate in the sickest
   observed-proxy quartile minus the healthiest - only informative
   dropout produces one), and *burstiness* (mean missing-run
   length). Diagnosis thresholds were fixed on pilot fleets
   (seeds 101-103), disjoint from the evaluation seeds, and frozen.
4. **Driver identification.** The profiler names the worst physical
   group under grouped dropout and the most health-coupled sensors
   under informative dropout, scored against the planted truth.

## 3. Prior art and the differentiator

- **Standard PHM robustness practice** (random-dropout stress tests
  in benchmark suites) assumes MCAR by construction. It cannot
  detect that it is wrong, because the audit and the deployment
  never meet.
- **TokaMark (2026)** benchmarks physically-grounded sensor failure
  for *plasma diagnostics*; correlated-failure PPO work is RL
  control. Neither targets RUL/PHM evaluation, and neither isolates
  failure-proximate *informative* missingness (MNAR tied to the
  label) as the mechanism that reverses imputation rankings.
- **Missing-data methodology** (Little's MCAR test, MAR/MNAR
  modelling) classifies mechanisms statistically but does not price
  them: it never reports what the wrong assumption costs a
  downstream predictor, per imputation strategy.

**Differentiator:** no existing tool plants matched-rate MCAR /
grouped / MNAR regimes over the same PHM trajectories, prices the
benchmark illusion per predictor x imputer cell, and diagnoses the
mechanism and its drivers from the mask against planted ground
truth.

## 4. The fleet simulator and the three regimes

- Seeded synthetic fleet (`dropout_profiler/generate.py`): each unit
  degrades on the health curve `h(t) = 1 - (t/T)^2.2`, fails at a
  known day `T` (90-150 days), and is observed daily on six sensors
  (vibration, temperature, pressure, current, acoustic, humidity).
  Readings are `baseline + gain * (1 - h) + noise` with a per-unit
  gain jitter; humidity is a near-null control sensor.
- Sensors sit on three physical groups: `bus_a` (vibration,
  temperature), `bus_b` (pressure, current), `zone_c` (acoustic,
  humidity).
- **iid:** every cell drops independently at p = 0.25.
- **grouped:** each group suffers outage episodes (start probability
  per idle day, geometric length, mean 5 days) during which all its
  sensors are dark together. `bus_b` is planted as the flaky bus
  (outage weight 1.5 vs 0.8 / 0.7).
- **informative:** sensor dropout probability is
  `0.07 + 2.30 * vulnerability * (1 - h)^3` (capped at 0.92), with
  vulnerability from 1.00 (vibration) down to 0.30 (humidity).
- Task: RUL regression at landmark days (every 5 days from day 10).
  Features are each sensor's imputed level and 10-day slope, plus
  age. Predictors: ridge, kNN, a depth-4 regression tree, and a
  single-sensor health-index regression. Imputers: fleet-mean fill,
  LOCF, linear interpolation, unit-mean fill.

## 5. The profiler

`dropout_profiler/profile.py` computes, from the masked panel alone:
overall / per-sensor / per-group missing rates; same-group and
cross-group co-missing lift; the degradation-proxy gradient (each
observed reading standardised by its sensor's fleet mean/sd, averaged
per unit-day); the life gradient (missing rate in the last life
quartile minus the first - the audit panel is run-to-failure, so
lifetimes are known); mean run length; and per-sensor proxy
gradients. Diagnosis: grouped if same-group lift >= 2.5 and >= 1.8x
the cross-group lift; else informative if the proxy gradient >=
0.12; else iid. Drivers: the highest-rate group, and the two sensors
with the steepest proxy gradients.

## 6. Headline results (measured)

6 paired seeds (201-206), 36 train / 20 test units per fleet.
Full tables in `results/`.

**Matched rates:** iid 0.249, grouped 0.250, informative 0.232.

**Diagnosis - profiler vs baselines:**

| method | accuracy |
|---|---|
| profiler (lift + proxy gradient) | **1.000** (1.00 per regime) |
| MAR screen (proxy gradient only, no grouped category) | 0.667 |
| always predict iid (the standard assumption) | 0.333 |

Driver identification accuracy: **1.000** - bus_b named as the worst
group in every grouped fleet; vibration + temperature named as the
top health-coupled sensors in every informative fleet. On the demo
fleet the signatures are unmistakable: grouped same-group lift 4.43
vs cross-group 0.93 with mean run length 4.71; informative proxy
gradient +0.309 and life gradient +0.569 vs -0.010 / -0.006 under iid.

**Robustness illusion gap** (RMSE in days, regime minus iid):

- The configuration an iid benchmark would choose (best imputer per
  predictor) degrades by **+0.44 days** under grouped dropout and
  **+0.49 days** under informative dropout. Interpolation, the usual
  iid winner, is genuinely robust - its cells move by at most
  +1.15 days.
- The illusion concentrates in the naive fills, and it is
  mechanism-specific. Under informative dropout the mean-fill cells
  degrade by **+1.93 days on average** and unit-mean by **+1.95**,
  while under grouped dropout the same fills are flat on average
  (+0.01 and +0.20 days). The worst cell in the grid - the single-sensor health index
  with mean fill - degrades by **+6.41 days** (23.08 -> 29.49 RMSE,
  +28%) under informative dropout, because mean fill replaces the
  hidden death-zone readings with fleet-average values and the
  health index reads them as a healthy machine.
- kNN is the honest exception: its mean-fill cells slightly
  *improve* under informative dropout (-0.22 days) - identical filled
  values make death-zone rows mutual neighbours, which happens to
  help a neighbourhood regressor. Reported as measured.

**Imputation-ranking reversal:** the best imputer under iid is not
the best under informative dropout in **37.5%** of (seed, predictor)
pairs (29.2% for iid -> grouped). It is never outright the worst
(0.0%) - the reversal is a reshuffle among the strong imputers plus
the collapse of the naive ones, not a perfect inversion.

Complete-data reference RMSE (mean): ridge 14.37, kNN 15.95,
tree 14.87, single-sensor 17.74 days.

## 7. Honest limits

- **Synthetic fleet.** Degradation is a smooth one-parameter curve
  and the informative mechanism is a clean cubic in true health;
  real sensor death is messier, partly recoverable, and sometimes
  caused by the maintenance the readings trigger. The measured
  numbers are simulator numbers - they validate the audit against
  planted ground truth because real panels do not record why a
  sensor went dark.
- **Perfect diagnosis is a property of these regimes.** The three
  mechanisms were designed to be separable by lift and gradient, and
  on 18 fleets they are. Real panels mix mechanisms (a flaky bus on
  a dying machine), and the profiler returns a single label; mixed
  panels are the obvious next step.
- **The chosen-configuration gap is small because interpolation is
  good.** The headline illusion lives in the naive fills and in the
  ranking reversal, not in a catastrophic failure of the best
  pipeline. A reader whose pipeline already interpolates should
  take the +0.49-day chosen-config gap, not the +6.41-day worst
  cell, as their number.
- **Diagnosis thresholds are calibrated, not derived.** They were
  fixed on pilot seeds 101-103 and frozen before evaluation; other
  fleets with different group counts or rates would need the pilot
  step repeated.
- **The life gradient uses lifetimes.** That is legitimate for a
  run-to-failure audit panel (the deployment setting this tool
  addresses), but the deployable-anywhere statistic is the proxy
  gradient, which is what the diagnosis rule actually uses.

## 8. How to run

```bash
pip install pytest streamlit   # streamlit only for the dashboard
python -m dropout_profiler demo                    # one paired seed, full grid + profile
python -m dropout_profiler evaluate --out results  # the headline evaluation
python scripts/reproduce_results.py                # re-run + drift guard
python notebooks/01_demo_run.py
python notebooks/02_evaluation_experiment.py
streamlit run app.py                               # dashboard (3 tabs)
pytest                                             # 62 tests
```

## 9. Repository layout

- `dropout_profiler/` - `models` (panel, groups, regime parameters),
  `generate` (fleet simulator + the three dropout regimes),
  `features` (imputers + landmark rows), `predictors` (ridge, kNN,
  regression tree, single-sensor health index), `profile` (mask
  statistics, diagnosis, drivers), `evaluate` (the grid + headline
  aggregation), `report` (text renderers), `cli`.
- `data/cases.json` - demo fleet summary + regime ground truth.
- `results/` - the stored headline evaluation (JSON, CSV, summary,
  HEADLINE).
- `notebooks/` - the demo run and the evaluation experiment as
  runnable scripts.
- `examples/` - a fast profile-only audit and a captured demo report.
- `scripts/` - the data builder and the reproduction/drift guard.
- `tests/` - 62 tests over the generator, regimes, imputers,
  predictors, profiler, evaluation, CLI, dashboard, and stored
  results.

Author: Amna Raheeba. License: MIT.
