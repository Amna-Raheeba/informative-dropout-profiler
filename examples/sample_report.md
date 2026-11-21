# Sample report

Output of `python -m dropout_profiler demo` (seed 7, paired fleets):

```
Informative Dropout Profiler - demo fleet (seed 7)
Train units: 36, test units: 20. Task: RUL regression at landmark days.

Complete-data reference RMSE (no dropout):
  ridge          14.41 days
  knn            15.12 days
  tree           15.00 days
  single_sensor  17.72 days

Regime: iid (planted driver: none)
  achieved missing rate 0.249 | diagnosis: iid (driver: none)
  same-group lift 1.03 | cross-group lift 0.98 | proxy gradient -0.010 | life gradient -0.006 | mean run length 1.32
  worst group: zone_c | top gradient sensors: temperature, vibration
  RMSE by imputer x predictor:
  imputer    ridge          knn            tree           single_sensor
  mean       15.26          16.82          16.09          22.87        
  locf       14.50          15.49          14.36          17.75        
  interp     14.27          15.23          14.30          16.90        
  unit_mean  15.36          16.76          15.60          23.02        

Regime: grouped (planted driver: group)
  achieved missing rate 0.236 | diagnosis: grouped (driver: group)
  same-group lift 4.43 | cross-group lift 0.93 | proxy gradient +0.004 | life gradient +0.012 | mean run length 4.71
  worst group: bus_b | top gradient sensors: vibration, temperature
  RMSE by imputer x predictor:
  imputer    ridge          knn            tree           single_sensor
  mean       15.51          16.38          15.97          22.07        
  locf       14.96          16.08          15.31          18.46        
  interp     14.52          15.23          14.97          17.53        
  unit_mean  15.86          16.50          16.07          22.69        

Regime: informative (planted driver: health)
  achieved missing rate 0.234 | diagnosis: informative (driver: health)
  same-group lift 2.21 | cross-group lift 2.18 | proxy gradient +0.309 | life gradient +0.569 | mean run length 2.12
  worst group: bus_a | top gradient sensors: vibration, temperature
  RMSE by imputer x predictor:
  imputer    ridge          knn            tree           single_sensor
  mean       15.92          15.53          16.54          28.15        
  locf       13.92          14.52          15.44          17.74        
  interp     13.86          14.71          14.84          17.18        
  unit_mean  15.95          15.39          16.40          28.02        
```
