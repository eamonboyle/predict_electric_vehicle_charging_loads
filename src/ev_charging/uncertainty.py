"""Distribution-free prediction intervals from validation residuals."""

from __future__ import annotations

import numpy as np


def residual_quantile_intervals(
    y_val: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_pred: np.ndarray,
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Add empirical residual quantiles to point predictions on the test set."""
    res = np.asarray(y_val, dtype=np.float64).ravel() - np.asarray(y_val_pred, dtype=np.float64).ravel()
    lo = np.quantile(res, q_low)
    hi = np.quantile(res, q_high)
    pred = np.asarray(y_test_pred, dtype=np.float64).ravel()
    return pred + lo, pred + hi
