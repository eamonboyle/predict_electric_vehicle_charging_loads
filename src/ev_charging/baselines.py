from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

from ev_charging.evaluate import regression_metrics


@dataclass
class BaselineBundle:
    models: dict[str, Any]
    metrics_test: dict[str, dict[str, float]]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> BaselineBundle:
        return joblib.load(path)


def fit_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    random_state: int = 42,
) -> BaselineBundle:
    specs: list[tuple[str, Any]] = [
        ("dummy_mean", DummyRegressor(strategy="mean")),
        ("dummy_median", DummyRegressor(strategy="median")),
        ("linear_regression", LinearRegression()),
        (
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.06,
                max_iter=200,
                random_state=random_state,
            ),
        ),
    ]
    models: dict[str, Any] = {}
    metrics_test: dict[str, dict[str, float]] = {}
    for name, est in specs:
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        models[name] = est
        metrics_test[name] = regression_metrics(y_test, pred)
    return BaselineBundle(models=models, metrics_test=metrics_test)


def permutation_importance_for_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int = 8,
    random_state: int = 42,
) -> list[tuple[str, float]]:
    nf = X_test.shape[1]
    names = list(feature_names)
    if len(names) != nf:
        names = [f"feature_{i}" for i in range(nf)]
    r = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    order = np.argsort(r.importances_mean)[::-1]
    return [(names[i], float(r.importances_mean[i])) for i in order if i < len(names)]
