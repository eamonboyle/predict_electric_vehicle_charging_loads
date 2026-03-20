"""FastAPI batch prediction from CSV upload."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, UploadFile

from ev_charging.config import load_config
from ev_charging.features import FeaturePreprocessor
from ev_charging.train_mlp import load_mlp_bundle, predict_mlp


def create_app(*, root: Path | None = None, config_rel: str | None = None) -> FastAPI:
    root = root or Path(os.environ.get("EV_CHARGING_ROOT", ".")).resolve()
    config_rel = config_rel or os.environ.get("EV_CHARGING_CONFIG", "configs/default.yaml")
    cfg = load_config(root / config_rel)
    out = cfg["outputs"]
    pre = FeaturePreprocessor.load(root / out["preprocessor_path"])
    model, _payload = load_mlp_bundle(root / out["model_path"])

    app = FastAPI(title="EV charging load prediction", version="0.1.0")
    app.state.root = root

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict/csv")
    async def predict_csv(
        file: UploadFile = File(...),
        sep: str = ";",
    ) -> dict[str, list[float]]:
        raw = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(raw)
            path = tmp.name
        try:
            df = pd.read_csv(path, sep=sep)
            X, _y = pre.transform(df)
            pred = predict_mlp(model, X)
        finally:
            Path(path).unlink(missing_ok=True)
        return {"predictions": [float(x) for x in pred]}

    return app


app = create_app()
