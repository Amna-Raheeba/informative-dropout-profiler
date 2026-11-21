"""Four small stdlib RUL regressors.

The panel is deliberately ordinary - these are the kinds of models a
PHM benchmark would actually contain:

- ``ridge``: ridge regression on standardised features (normal
  equations solved by Gaussian elimination).
- ``knn``: k-nearest-neighbour regression on standardised features.
- ``tree``: a depth-limited regression tree (greedy variance
  reduction over quantile candidate thresholds).
- ``single_sensor``: univariate linear regression on the single
  sensor level most correlated with RUL in training - the classic
  health-index baseline.

Every predictor exposes fit(X, y) / predict(X) and is deterministic.
"""

import math

from .models import SENSOR_NAMES

RIDGE_LAMBDA = 1.0
KNN_K = 7
TREE_MAX_DEPTH = 4
TREE_MIN_LEAF = 25


def rmse(predicted, actual):
    if not actual:
        return float("nan")
    return math.sqrt(
        sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(actual)
    )


def _mean(values):
    return sum(values) / len(values)


def _std(values, mean):
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) if var > 0 else 1.0


def _solve(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pivot_value = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col] / pivot_value
                for c in range(col, n + 1):
                    M[r][c] -= factor * M[col][c]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
            for i in range(n)]


class _Standardiser(object):
    def __init__(self, X):
        self.means = []
        self.stds = []
        for j in range(len(X[0])):
            col = [row[j] for row in X]
            m = _mean(col)
            self.means.append(m)
            self.stds.append(_std(col, m))

    def apply(self, X):
        return [
            [(row[j] - self.means[j]) / self.stds[j]
             for j in range(len(row))]
            for row in X
        ]


class RidgeRegressor(object):
    def fit(self, X, y):
        self._std = _Standardiser(X)
        Z = self._std.apply(X)
        self._y_mean = _mean(y)
        d = len(X[0])
        A = [[0.0] * d for _ in range(d)]
        b = [0.0] * d
        for row, target in zip(Z, y):
            centred = target - self._y_mean
            for i in range(d):
                b[i] += row[i] * centred
                for j in range(d):
                    A[i][j] += row[i] * row[j]
        for i in range(d):
            A[i][i] += RIDGE_LAMBDA
        self._w = _solve(A, b)
        return self

    def predict(self, X):
        Z = self._std.apply(X)
        out = []
        for row in Z:
            value = self._y_mean + sum(w * v for w, v in zip(self._w, row))
            out.append(max(0.0, value))
        return out


class KnnRegressor(object):
    def fit(self, X, y):
        self._std = _Standardiser(X)
        self._X = self._std.apply(X)
        self._y = list(y)
        return self

    def predict(self, X):
        Z = self._std.apply(X)
        out = []
        for row in Z:
            dists = []
            for i, train_row in enumerate(self._X):
                d2 = 0.0
                for a, b in zip(row, train_row):
                    diff = a - b
                    d2 += diff * diff
                dists.append((d2, i))
            dists.sort()
            neighbours = dists[:KNN_K]
            out.append(max(0.0, _mean([self._y[i] for _, i in neighbours])))
        return out


class _TreeNode(object):
    __slots__ = ("value", "feature", "threshold", "left", "right")

    def __init__(self, value):
        self.value = value
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None


def _sse(values, mean):
    return sum((v - mean) ** 2 for v in values)


class TreeRegressor(object):
    def fit(self, X, y):
        self._root = self._build(X, y, list(range(len(y))), 0)
        return self

    def _build(self, X, y, idx, depth):
        values = [y[i] for i in idx]
        node = _TreeNode(_mean(values))
        if depth >= TREE_MAX_DEPTH or len(idx) < 2 * TREE_MIN_LEAF:
            return node
        base_sse = _sse(values, node.value)
        if base_sse <= 0.0:
            return node
        best = None
        for j in range(len(X[0])):
            col = sorted(X[i][j] for i in idx)
            candidates = []
            for q in (0.2, 0.35, 0.5, 0.65, 0.8):
                candidates.append(col[int(q * (len(col) - 1))])
            for threshold in candidates:
                left = [i for i in idx if X[i][j] <= threshold]
                right = [i for i in idx if X[i][j] > threshold]
                if len(left) < TREE_MIN_LEAF or len(right) < TREE_MIN_LEAF:
                    continue
                left_mean = _mean([y[i] for i in left])
                right_mean = _mean([y[i] for i in right])
                gain = (base_sse
                        - _sse([y[i] for i in left], left_mean)
                        - _sse([y[i] for i in right], right_mean))
                if best is None or gain > best[0]:
                    best = (gain, j, threshold, left, right)
        if best is None or best[0] <= 1e-9:
            return node
        _, j, threshold, left, right = best
        node.feature = j
        node.threshold = threshold
        node.left = self._build(X, y, left, depth + 1)
        node.right = self._build(X, y, right, depth + 1)
        return node

    def predict(self, X):
        out = []
        for row in X:
            node = self._root
            while node.feature is not None:
                if row[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            out.append(max(0.0, node.value))
        return out


class SingleSensorRegressor(object):
    """Univariate linear fit on the best-correlated sensor level."""

    def fit(self, X, y):
        y_mean = _mean(y)
        best = None
        for j in range(len(SENSOR_NAMES)):
            col = [row[j] for row in X]
            x_mean = _mean(col)
            num = sum((a - x_mean) * (b - y_mean) for a, b in zip(col, y))
            den = sum((a - x_mean) ** 2 for a in col)
            corr = num / math.sqrt(den * _sse(y, y_mean)) if den > 0 else 0.0
            if best is None or abs(corr) > abs(best[0]):
                best = (corr, j, x_mean, num / den if den > 0 else 0.0)
        _, self._j, self._x_mean, self._slope = best
        self._y_mean = y_mean
        self.sensor = SENSOR_NAMES[self._j]
        return self

    def predict(self, X):
        return [
            max(0.0, self._y_mean + self._slope * (row[self._j] - self._x_mean))
            for row in X
        ]


PREDICTOR_CLASSES = {
    "ridge": RidgeRegressor,
    "knn": KnnRegressor,
    "tree": TreeRegressor,
    "single_sensor": SingleSensorRegressor,
}


def fit_predictor(name, X, y):
    model = PREDICTOR_CLASSES[name]()
    return model.fit(X, y)
