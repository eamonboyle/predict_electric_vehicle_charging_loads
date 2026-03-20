from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from ev_charging.evaluate import regression_metrics


def build_mlp(n_in: int, hidden1: int, hidden2: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden1),
        nn.ReLU(),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(),
        nn.Linear(hidden2, 1),
    )


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    *,
    epochs: int,
    lr: float,
    hidden1: int,
    hidden2: int,
    weight_decay: float = 0.0,
    torch_seed: int = 42,
    early_stopping_patience: int = 0,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    device = device or torch.device("cpu")
    torch.manual_seed(torch_seed)
    n_in = X_train.shape[1]
    model = build_mlp(n_in, hidden1, hidden2).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)
    Xv_t = yv_t = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        Xv_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        yv_t = torch.tensor(y_val, dtype=torch.float32, device=device).reshape(-1, 1)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    bad_epochs = 0
    history: list[tuple[int, float, float | None]] = []

    for epoch in range(epochs):
        model.train()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = float(loss.detach().cpu())

        val_loss: float | None = None
        if Xv_t is not None and yv_t is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xv_t), yv_t).cpu())
            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if early_stopping_patience > 0 and bad_epochs >= early_stopping_patience:
                break

        if epoch % max(1, epochs // 9) == 0 or epoch == epochs - 1:
            history.append((epoch, train_loss, val_loss))

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"history": history, "best_val_mse": best_val if best_state is not None else None}
    return model, meta


def predict_mlp(model: nn.Module, X: np.ndarray, device: str | torch.device | None = None) -> np.ndarray:
    device = device or torch.device("cpu")
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32, device=device)
        out = model(t).cpu().numpy().ravel()
    return out


def save_mlp_bundle(
    path: str | Path,
    model: nn.Module,
    *,
    hidden1: int,
    hidden2: int,
    n_features: int,
    torch_seed: int,
    train_meta: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "hidden1": hidden1,
        "hidden2": hidden2,
        "n_features": n_features,
        "torch_seed": torch_seed,
        "train_meta": train_meta or {},
    }
    torch.save(payload, path)


def load_mlp_bundle(path: str | Path, device: str | torch.device | None = None) -> tuple[nn.Module, dict[str, Any]]:
    device = device or torch.device("cpu")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    n_in = int(payload["n_features"])
    model = build_mlp(n_in, int(payload["hidden1"]), int(payload["hidden2"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def mlp_test_metrics(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    pred = predict_mlp(model, X_test)
    return regression_metrics(y_test, pred)
