"""Streamlit UI: upload a feature CSV and append predictions."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from ev_charging.config import load_config
from ev_charging.features import FeaturePreprocessor
from ev_charging.train_mlp import load_mlp_bundle, predict_mlp


@st.cache_resource
def _load_predictor(root: str, config_rel: str):
    root_p = Path(root).resolve()
    cfg = load_config(root_p / config_rel)
    out = cfg["outputs"]
    pre = FeaturePreprocessor.load(root_p / out["preprocessor_path"])
    model, payload = load_mlp_bundle(root_p / out["model_path"])
    return pre, model


st.set_page_config(page_title="EV charging kWh", layout="wide")
st.title("EV charging load prediction")

root = os.environ.get("EV_CHARGING_ROOT", ".")
config_rel = os.environ.get("EV_CHARGING_CONFIG", "configs/default.yaml")

try:
    pre, model = _load_predictor(root, config_rel)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load model or preprocessor. Train first: `python -m ev_charging train`. ({e})")
    st.stop()

f = st.file_uploader("Session CSV (same feature columns as training)", type=["csv"])
sep = st.text_input("Delimiter", value=";")

if f is not None and st.button("Predict"):
    df = pd.read_csv(f, sep=sep or ";")
    X, _y = pre.transform(df)
    pred = predict_mlp(model, X)
    df = df.copy()
    df["predicted_kWh"] = pred
    st.dataframe(df, use_container_width=True)
