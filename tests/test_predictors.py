import random

from dropout_profiler.models import SENSOR_NAMES
from dropout_profiler.predictors import fit_predictor, rmse


def test_rmse_known_values():
    assert rmse([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert abs(rmse([0.0, 0.0], [3.0, 4.0]) - (25.0 / 2.0) ** 0.5) < 1e-9


def _linear_data(n=200, seed=3):
    rng = random.Random(seed)
    X = []
    y = []
    for _ in range(n):
        row = [rng.uniform(-2, 2) for _ in range(13)]
        X.append(row)
        y.append(30.0 + 4.0 * row[0] - 2.0 * row[1] + 0.5 * row[12])
    return X, y


def test_ridge_recovers_linear_relation():
    X, y = _linear_data()
    model = fit_predictor("ridge", X, y)
    assert rmse(model.predict(X), y) < 1.0


def test_knn_fits_smooth_relation():
    X, y = _linear_data()
    model = fit_predictor("knn", X, y)
    assert rmse(model.predict(X), y) < 3.0


def test_tree_beats_mean_on_step_function():
    rng = random.Random(4)
    X = []
    y = []
    for _ in range(300):
        row = [rng.uniform(0, 1) for _ in range(13)]
        X.append(row)
        y.append(10.0 if row[0] > 0.5 else 40.0)
    model = fit_predictor("tree", X, y)
    mean_pred = [sum(y) / len(y)] * len(y)
    assert rmse(model.predict(X), y) < 0.5 * rmse(mean_pred, y)


def test_single_sensor_picks_the_driving_sensor():
    rng = random.Random(5)
    X = []
    y = []
    for _ in range(200):
        levels = [rng.uniform(0, 3) for _ in SENSOR_NAMES]
        row = levels + [0.0] * len(SENSOR_NAMES) + [50.0]
        X.append(row)
        y.append(100.0 - 20.0 * levels[2])  # pressure drives RUL
    model = fit_predictor("single_sensor", X, y)
    assert model.sensor == "pressure"
    assert rmse(model.predict(X), y) < 2.0


def test_all_predictors_clip_negative_rul():
    X, y = _linear_data(60)
    far = [[100.0] * 13]
    for name in ("ridge", "knn", "tree", "single_sensor"):
        model = fit_predictor(name, X, y)
        assert model.predict(far)[0] >= 0.0


def test_predictors_are_deterministic():
    X, y = _linear_data(80)
    for name in ("ridge", "knn", "tree", "single_sensor"):
        a = fit_predictor(name, X, y).predict(X[:5])
        b = fit_predictor(name, X, y).predict(X[:5])
        assert a == b
