from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = np.abs(y_true) > 1e-6
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("nan")
    return {"mse": float(mse), "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    *,
    title: str = "Predicted vs actual",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    r = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r, bins=40, color="steelblue", edgecolor="white")
    ax.set_title("Residuals (actual − predicted)")
    ax.set_xlabel("kWh")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def metrics_table_row(model_name: str, metrics: dict[str, float]) -> dict[str, Any]:
    row = {"model": model_name, **metrics}
    return row


def append_metrics_csv(path: str | Path, row: dict[str, Any]) -> None:
    from datetime import datetime, timezone

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**row, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined.to_csv(path, index=False)
